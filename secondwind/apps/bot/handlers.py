import asyncio
import hashlib
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any
from collections.abc import Awaitable, Callable
import logging
import os
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache

from telegram import (BotCommand, BotCommandScopeChat, InlineKeyboardButton, InlineKeyboardMarkup, InputFile, Message,
                      Update)
from telegram.error import BadRequest
from telegram.ext import ContextTypes
from telegram.helpers import escape_markdown

from secondwind.adapters.llm.claude import ClaudeLLM
from secondwind.adapters.warehouse.bigquery_reader import BigQueryReader
from secondwind.core.assistant import ASK_PROMPT, split_scope, COACH_PROMPT, HEALTH_PROMPT, NUTRITION_PROMPT, Refused, analyze
from secondwind.core.freshness import stale_warnings
from secondwind.core.guard import (budget_allowed, forget_message, is_repeat, model_allowed, record_message,
                                   record_model_call, record_spend)
from secondwind.core.router import Route, is_red_flag, route as route_message
from secondwind.core import answer_cache, dialog
from secondwind.core.assistant import FORMAT_NOTE, PERSONA
from secondwind.core.scope import ERROR as SCOPE_ERROR, Verdict, check, scope_route
from secondwind.core.ports import LLM, DataReader, StateStore
from secondwind.core import analytics
from secondwind.apps.bot import charts, tenancy
from secondwind.core.athlete import DISTANCES, PROFILE_FIELDS, athlete_note, compute_zones
from secondwind.core.estimates import readiness, training_status
from secondwind.core.i18n import SUPPORTED, resolve, t
from secondwind.core.models import DEFAULT_MODE
from secondwind.adapters.state.firestore import FirestoreStore
from secondwind.apps.bot.user_ui import (apply_setting, ask_text, distance_keyboard, handle_input, render_goal, render_profile,
                     render_settings)
from secondwind.apps.bot.tg_format import html_to_plain, md_to_html, split_message
from secondwind.apps.bot.user_ui import _zones_title

logger = logging.getLogger(__name__)

SOURCE_LABEL = {"garmin": "Garmin", "strava": "Strava"}
# Questions a user may send to the AI model per hour (filter, architecture 2.1).
MODEL_CALLS_PER_HOUR = int(os.environ.get("MODEL_CALLS_PER_HOUR", "20"))


def _daily_budget() -> float:
    """AI spending per user per UTC day, USD (DAILY_BUDGET_USD, checked at startup)."""
    return float(os.environ.get("DAILY_BUDGET_USD", "") or 1.0)


CONTEXT_LANG = "en"  # data handed to Claude is formatted in English; Claude answers in the user's language

# Commands in the Telegram "Menu" list, in display order. /start stays hidden (Telegram sends it itself).
MENU_COMMANDS = ["menu", "today", "week", "month", "compare", "records", "charts", "coach", "health", "nutrition",
                 "goal", "profile", "settings", "status"]
# Inline buttons under /start and /menu, two per row.
MENU_BUTTONS = [["today", "week"], ["month", "compare"], ["records", "charts"], ["coach", "health"],
                ["nutrition", "goal"], ["profile", "settings"], ["status"]]


def _bq(update: Update) -> DataReader:
    """This chat's user's data."""
    return BigQueryReader(os.environ["GCP_PROJECT"], os.environ["BQ_DATASET"], _uid(update))


def _enabled_sources():
    return [s.strip() for s in os.environ.get("DATA_SOURCES", "").split(",") if s.strip()]


def _owner() -> str:
    return tenancy.owner_chat()


def _allowed(update: Update) -> bool:
    """Only chats that map to a user may use the bot (it exposes health data and spends AI credits).
    In solo mode that is the owner's chat, and only once OWNER_USER_ID is set."""
    chat = update.effective_chat
    return tenancy.user_for_chat(chat.id if chat is not None else None) is not None


@lru_cache(maxsize=1)
def _store() -> StateStore:
    return FirestoreStore(os.environ["GCP_PROJECT"])


@lru_cache(maxsize=1)
def _llm() -> LLM:
    return ClaudeLLM()


def _uid(update: Update) -> str:
    """The internal user_id of this chat (callers have passed _guard, so both exist)."""
    chat = update.effective_chat
    uid = tenancy.user_for_chat(chat.id if chat is not None else None)
    if uid is None:
        raise PermissionError("chat is not linked to a user")
    return uid


def _user(update: Update) -> dict:
    """The user's stored state; {} if storage is unavailable (the bot keeps working without it)."""
    chat = update.effective_chat
    uid = tenancy.user_for_chat(chat.id) if chat is not None else None
    if uid is None:
        return {}
    try:
        return _store().get_user(uid)
    except Exception as e:  # noqa: BLE001
        logger.error("state read failed: %s", e)
        return {}


def _lang(update: Update, user: dict | None = None) -> str:
    """Language chosen in /settings, else BOT_LANGUAGE (set at deploy), else the Telegram app language."""
    chosen = ((user if user is not None else _user(update)).get("settings") or {}).get("lang")
    if chosen in SUPPORTED:
        return chosen
    forced = os.environ.get("BOT_LANGUAGE", "").strip()
    if forced:
        return resolve(forced)
    tg_user = update.effective_user
    return resolve(tg_user.language_code if tg_user else None)


def _md(v) -> str:
    return escape_markdown(str(v), version=1)


def _hm(seconds, lang) -> str:
    if not seconds:
        return "—"
    h, m = divmod(int(seconds) // 60, 60)
    return t(lang, "fmt.hm", h=h, m=m)


def _fmt_activity(a, lang):
    dist = (a.get("distance_meters") or 0) / 1000
    dur = (a.get("duration_seconds") or 0) // 60
    pace = a.get("avg_pace_sec_per_km")
    pace_str = t(lang, "act.pace", m=int(pace // 60), s=int(pace % 60)) if pace else "—"
    src = SOURCE_LABEL.get(a.get("source"), a.get("source") or "—")
    lines = [
        f"*{_md(a.get('name') or '—')}* ({_md(a.get('type') or '—')}, {src})",
        t(lang, "act.line2", date=a.get("activity_date"), km=round(dist, 1), min=dur),
        t(lang, "act.line3", pace=pace_str, avg=a.get("avg_heart_rate") or "—", max=a.get("max_heart_rate") or "—"),
        t(lang, "act.line4", elev=round(a.get("elevation_gain") or 0), cal=a.get("calories") or "—"),
    ]
    extra = []
    if a.get("aerobic_te") is not None:
        extra.append(t(lang, "act.te", a=f"{a['aerobic_te']:.1f}", an=f"{a.get('anaerobic_te') or 0:.1f}"))
    if a.get("training_load") is not None:
        extra.append(t(lang, "act.load", v=round(a["training_load"])))
    if a.get("vo2max") is not None:
        extra.append(f"VO2max {a['vo2max']:.0f}")
    if extra:
        lines.append(" | ".join(extra))
    return "\n".join(lines)


def _fmt_health(h, lang):
    parts = [f"📅 {h['date']}"]
    if h.get("sleep_seconds"):
        parts.append(t(lang, "health.sleep", total=_hm(h["sleep_seconds"], lang),
                       deep=_hm(h.get("deep_sleep_seconds"), lang), rem=_hm(h.get("rem_sleep_seconds"), lang),
                       score=h.get("sleep_score") or "—"))
    if h.get("hrv_last_night"):
        norm = (t(lang, "health.hrv_norm", lo=h["hrv_baseline_low"], hi=h["hrv_baseline_high"])
                if h.get("hrv_baseline_low") and h.get("hrv_baseline_high") else "")
        parts.append(t(lang, "health.hrv", v=h["hrv_last_night"], w=h.get("hrv_weekly_avg") or "—",
                       norm=norm, status=h.get("hrv_status") or "—"))
    vitals = []
    if h.get("resting_hr"):
        vitals.append(t(lang, "health.rhr", v=h["resting_hr"]))
    if h.get("stress_avg") is not None:
        vitals.append(t(lang, "health.stress", v=h["stress_avg"]))
    if h.get("body_battery_high") is not None:
        vitals.append(f"BB {h.get('body_battery_low') or '—'}→{h['body_battery_high']}")
    if h.get("steps"):
        vitals.append(t(lang, "health.steps", v=h["steps"]))
    if vitals:
        parts.append("📊 " + " | ".join(vitals))
    return "\n".join(parts)


FOCUS = {"focus_low_aerobic_28d": "low_aerobic", "focus_high_aerobic_28d": "high_aerobic",
         "focus_anaerobic_28d": "anaerobic"}


def _fmt_load(l, lang):
    if not l:
        return t(lang, "load.none")
    ratio = l.get("load_ratio")
    ratio_str = (t(lang, "load.ratio", r=f"{ratio:.2f}", zone=l["load_ratio_zone"])
                 if ratio is not None and l.get("load_ratio_zone") else t(lang, "load.short_history"))
    lines = [t(lang, "load.line", acute=round(l.get("acute_load") or 0),
               chronic=round(l.get("chronic_load") or 0), ratio=ratio_str)]
    total = sum(l.get(k) or 0 for k in FOCUS)
    if total:
        lines.append(t(lang, "load.focus", parts=", ".join(
            f"{t(lang, 'load.focus.' + name)} {round(100 * (l.get(k) or 0) / total)}%" for k, name in FOCUS.items())))
    if l.get("vo2max") is not None:
        old = l.get("vo2max_28d_ago")
        lines.append(t(lang, "load.vo2", v=f"{l['vo2max']:.1f}")
                     + (t(lang, "load.vo2_trend", d=f"{l['vo2max'] - old:+.1f}") if old is not None else ""))
    return "\n".join(lines)


def _estimates(health, load, lang) -> str:
    return (readiness(health, load, lang=lang).text(t(lang, "est.readiness")) + "\n"
            + training_status(load, health, lang=lang).text(t(lang, "est.status")))


CONTEXT_WORKERS = 9   # the context queries are independent: run them at the same time


def _context(bq: DataReader) -> str:
    """Data block for Claude: what we have, plus explicit gaps. Always in English."""
    lang = CONTEXT_LANG
    today = date.today()
    month_start = today.replace(day=1)
    prev_end = month_start - timedelta(days=1)
    queries: dict[str, Callable[[], Any]] = {
        "last": bq.get_last_activity, "acts": bq.get_week_activities,
        "prev": lambda: bq.get_week_activities(weeks_back=1), "health": lambda: bq.get_health_last_days(14),
        "loads": lambda: bq.get_load_last_days(1), "status": bq.get_source_status,
        "month": lambda: bq.get_period(month_start, today),
        "prev_month": lambda: bq.get_period(prev_end.replace(day=1), prev_end),
        "records": lambda: bq.get_records(today - timedelta(days=365))}
    with ThreadPoolExecutor(CONTEXT_WORKERS) as pool:
        q: dict[str, Future[Any]] = {name: pool.submit(fn) for name, fn in queries.items()}
    last, acts, prev, health, loads = (q[k].result() for k in ("last", "acts", "prev", "health", "loads"))
    load = loads[-1] if loads else None
    warn = stale_warnings(q["status"].result(), _enabled_sources(), lang)
    try:
        month = analytics.month_note(q["month"].result(), q["prev_month"].result(), today)
        records = analytics.records_note(q["records"].result())
    except Exception as e:  # noqa: BLE001 - extra context must not block the answer
        logger.error("analytics context failed: %s", e)
        month = records = "unavailable"
    return (
        "=== LAST WORKOUT ===\n" + (_fmt_activity(last, lang) if last else "no data") + "\n\n"
        "=== WORKOUTS: THIS WEEK ===\n" + ("\n\n".join(_fmt_activity(a, lang) for a in acts) or "none") + "\n\n"
        "=== WORKOUTS: LAST WEEK ===\n" + ("\n\n".join(_fmt_activity(a, lang) for a in prev) or "none") + "\n\n"
        "=== HEALTH BY DAY (14 days: sleep, HRV, resting HR, stress, Body Battery) ===\n"
        + ("\n".join(_fmt_health(h, lang) for h in health) or "no data") + "\n\n"
        "=== TRAINING LOAD (computed by Second Wind from Garmin data) ===\n" + _fmt_load(load, lang) + "\n\n"
        "=== SECOND WIND ESTIMATES (NOT Garmin metrics; call them our estimate) ===\n"
        + _estimates(health, load, lang) + "\n\n"
        "=== MONTH ===\n" + month + "\n\n"
        "=== PERSONAL RECORDS (best efforts within runs, Garmin) ===\n" + records + "\n\n"
        + ("=== DATA FRESHNESS ===\n" + "\n".join(warn) if warn else "")
    )


async def _io(fn, *args, **kw):
    """Run a blocking call (BigQuery, Firestore, the AI model, charts) in a thread, so the event
    loop keeps serving other messages meanwhile."""
    return await asyncio.to_thread(fn, *args, **kw)


async def _alang(update: Update) -> str:
    return _lang(update, await _io(_user, update))


async def _guard(update: Update) -> bool:
    if _allowed(update):
        return True
    logger.warning("rejected chat %s", update.effective_chat.id if update.effective_chat else "?")
    return False


def _message(update: Update) -> Message:
    msg = update.effective_message
    if msg is None:
        raise ValueError("update without a message")
    return msg


async def _reply(update: Update, text: str, **kw):
    return await _message(update).reply_text(text, **kw)


# ---------- menu ----------

def _menu_markup(lang: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton(t(lang, f"btn.{c}"), callback_data=f"cmd:{c}") for c in row]
                                 for row in MENU_BUTTONS])


async def setup_menu(bot) -> None:
    """Register the command list shown under Telegram's "Menu" button, only in the owner's chat."""
    owner = _owner()
    if not owner:
        logger.warning("TELEGRAM_CHAT_ID is not set; command menu not registered")
        return
    scope = BotCommandScopeChat(int(owner))
    # A language fixed in /settings or by BOT_LANGUAGE applies whatever the Telegram app language is.
    try:
        uid = tenancy.owner_user_id()
        user = await _io(lambda: _store().get_user(uid)) if uid else {}
        chosen = (user.get("settings") or {}).get("lang")
    except Exception as e:  # noqa: BLE001
        logger.error("state read failed: %s", e)
        chosen = None
    forced = chosen if chosen in SUPPORTED else (resolve(os.environ["BOT_LANGUAGE"])
                                                 if os.environ.get("BOT_LANGUAGE", "").strip() else None)
    for lang in SUPPORTED:
        shown = forced or lang
        commands = [BotCommand(c, t(shown, f"cmd.{c}")) for c in MENU_COMMANDS]
        # The English list (no language_code) is also the fallback for any other Telegram language.
        await bot.set_my_commands(commands, scope=scope, language_code=None if lang == "en" else lang)
    await _set_names(bot)


async def _set_names(bot) -> None:
    """Bot display name per Telegram language (e.g. "Вита · Второе дыхание"); English is the fallback.
    Only changed when different: setMyName is strictly rate-limited."""
    for lang in SUPPORTED:
        code = None if lang == "en" else lang
        name = t(lang, "bot.name")
        try:
            if (await bot.get_my_name(language_code=code)).name != name:
                await bot.set_my_name(name, language_code=code)
        except Exception as e:  # noqa: BLE001 - a name is cosmetic; never break startup over it
            logger.warning("bot name for %s not set: %s", lang, e)


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE | None):
    if not await _guard(update):
        return
    lang = await _alang(update)
    await _reply(update, t(lang, "menu.title"), reply_markup=_menu_markup(lang))


cmd_menu = cmd_start


async def on_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE | None):
    """All inline buttons: cmd:<command>, set:<key>:<value>, prof:..., goal:..."""
    query = update.callback_query
    if query is None:
        return
    await query.answer()
    if not await _guard(update):
        return
    kind, _, rest = (query.data or "").partition(":")
    if kind == "cmd" and rest in COMMANDS:
        await COMMANDS[rest](update, ctx)
    elif kind == "set":
        await _on_setting(update, *rest.partition(":")[::2])
    elif kind == "prof":
        await _on_profile_button(update, rest)
    elif kind == "goal":
        await _on_goal_button(update, rest)
    elif kind == "cmp" and rest in analytics.COMPARE_SPANS:
        await _send_compare(update, rest)
    elif kind == "chart" and rest in charts.CHARTS:
        await _send_chart(update, rest)
    elif kind == "more":
        await _on_more(update, rest)


async def _on_more(update: Update, entry_id: str):
    """The same question again, in detail, one model tier up; the button is used once."""
    query = update.callback_query
    try:
        if query is not None:
            await query.edit_message_reply_markup(reply_markup=None)
    except BadRequest as e:
        logger.info("button not removed: %s", e)
    entries = await _io(_recent_dialog, update)
    i = next((n for n, e in enumerate(entries) if e.get("id") == entry_id), None)
    if i is None:
        await _reply(update, t(await _alang(update), "more.expired"))
        return
    earlier = dialog.block(entries[:i]) if i else None
    await _ai_reply(update, ASK_PROMPT, "more", "chat", detail=True, question=entries[i]["q"], dialog=earlier)


# ---------- analytics without AI ----------

async def cmd_records(update: Update, ctx: ContextTypes.DEFAULT_TYPE | None):
    if not await _guard(update):
        return
    lang = await _alang(update)
    rows = await _io(lambda: _bq(update).get_records(date.today() - timedelta(days=365)))
    await _reply(update, analytics.render_records(lang, rows), parse_mode="HTML")


async def cmd_month(update: Update, ctx: ContextTypes.DEFAULT_TYPE | None):
    if not await _guard(update):
        return
    lang = await _alang(update)
    today = date.today()
    prev_end = today.replace(day=1) - timedelta(days=1)

    def build():
        bq = _bq(update)
        return analytics.render_month(lang, today, bq.get_period(today.replace(day=1), today),
                                      bq.get_period(prev_end.replace(day=1), prev_end))
    text = await _io(build)
    await _reply(update, text, parse_mode="HTML")


async def cmd_compare(update: Update, ctx: ContextTypes.DEFAULT_TYPE | None):
    if not await _guard(update):
        return
    lang = await _alang(update)
    kb = InlineKeyboardMarkup([[InlineKeyboardButton(t(lang, f"cmp.btn.{s}"), callback_data=f"cmp:{s}")
                                for s in analytics.COMPARE_SPANS]])
    await _reply(update, t(lang, "cmp.pick"), reply_markup=kb)


async def _send_compare(update: Update, span: str):
    lang = await _alang(update)
    today = date.today()
    (a0, a1), (b0, b1) = analytics.periods(span, today)

    def build():
        bq = _bq(update)
        return analytics.render_compare(lang, span, today, bq.get_period(a0, a1), bq.get_period(b0, b1))
    text = await _io(build)
    await _reply(update, text, parse_mode="HTML")


async def cmd_charts(update: Update, ctx: ContextTypes.DEFAULT_TYPE | None):
    if not await _guard(update):
        return
    lang = await _alang(update)
    names = list(charts.CHARTS)
    kb = InlineKeyboardMarkup([[InlineKeyboardButton(t(lang, f"chart.btn.{c}"), callback_data=f"chart:{c}")
                                for c in names[i:i + 2]] for i in range(0, len(names), 2)])
    await _reply(update, t(lang, "chart.pick"), reply_markup=kb)


async def _send_chart(update: Update, name: str):
    lang = await _alang(update)
    png = await _io(lambda: charts.render(name, lang, _bq(update)))
    await _message(update).reply_photo(InputFile(png, filename=f"{name}.png"))


# ---------- settings, profile, goal ----------

async def _show(update: Update, text: str, kb, edit: bool = False):
    """Show a screen: edit the message with the pressed button, or send a new one."""
    if edit and update.callback_query:
        try:
            await update.callback_query.edit_message_text(text, parse_mode="HTML", reply_markup=kb)
            return
        except BadRequest as e:  # e.g. "message is not modified"
            logger.info("edit skipped: %s", e)
            return
    await _reply(update, text, parse_mode="HTML", reply_markup=kb)


async def _stored(update: Update, lang: str, action) -> bool:
    """Run a state write; tell the user if storage is down."""
    try:
        await _io(action)
        return True
    except Exception as e:  # noqa: BLE001
        logger.error("state write failed: %s", e)
        await _reply(update, t(lang, "storage.error"))
        return False


def _athlete_inputs(update: Update) -> dict:
    """Watch zones, threshold trend and Garmin resting HR; {} if BigQuery can't answer."""
    try:
        return _bq(update).get_athlete_inputs()
    except Exception as e:  # noqa: BLE001
        logger.error("athlete inputs read failed: %s", e)
        return {}


def _month() -> str:
    return date.today().strftime("%Y-%m")


async def cmd_settings(update: Update, ctx: ContextTypes.DEFAULT_TYPE | None, edit: bool = False):
    if not await _guard(update):
        return
    user = await _io(_user, update)
    lang = _lang(update, user)
    try:
        usage = await _io(lambda: _store().get_usage(_uid(update), _month()))
    except Exception as e:  # noqa: BLE001
        logger.error("usage read failed: %s", e)
        usage = {}
    await _show(update, *render_settings(lang, user.get("settings") or {}, usage), edit=edit)


async def _on_setting(update: Update, key: str, value: str):
    lang = await _alang(update)
    ok = await _stored(update, lang, lambda: apply_setting(_store(), _uid(update), key, value))
    if ok:
        if key == "lang":
            await setup_menu(update.get_bot())
        await cmd_settings(update, None, edit=True)


async def cmd_profile(update: Update, ctx: ContextTypes.DEFAULT_TYPE | None, edit: bool = False):
    if not await _guard(update):
        return
    user = await _io(_user, update)
    inputs = await _io(_athlete_inputs, update)
    await _show(update, *render_profile(_lang(update, user), user.get("profile") or {}, inputs), edit=edit)


async def _on_profile_button(update: Update, rest: str):
    lang = await _alang(update)
    action, _, field = rest.partition(":")
    if action == "clear":
        if await _stored(update, lang, lambda: _store().clear_profile(_uid(update))):
            await _reply(update, t(lang, "prof.cleared"))
            await cmd_profile(update, None)
    elif action == "edit" and field in PROFILE_FIELDS:
        pending = {"kind": "profile", "field": field}
        if await _stored(update, lang, lambda: _store().set_pending(_uid(update), pending)):
            await _reply(update, ask_text(lang, pending))


async def cmd_goal(update: Update, ctx: ContextTypes.DEFAULT_TYPE | None, edit: bool = False):
    if not await _guard(update):
        return
    lang = await _alang(update)
    try:
        goal = await _io(lambda: _store().get_goal(_uid(update)))
    except Exception as e:  # noqa: BLE001
        logger.error("goal read failed: %s", e)
        goal = None
    await _show(update, *render_goal(lang, goal, date.today()), edit=edit)


async def _on_goal_button(update: Update, rest: str):
    lang = await _alang(update)
    action, _, value = rest.partition(":")
    uid = _uid(update)
    if action == "new":
        await _reply(update, t(lang, "goal.ask.distance"), reply_markup=distance_keyboard(lang))
    elif action == "delete":
        if await _stored(update, lang, lambda: _store().delete_goal(uid)):
            await _reply(update, t(lang, "goal.deleted"))
    elif action == "dist":
        km = DISTANCES.get(value)
        pending = ({"kind": "goal", "step": "date", "data": {"distance_km": km}} if km
                   else {"kind": "goal", "step": "distance_other", "data": {}})
        if await _stored(update, lang, lambda: _store().set_pending(uid, pending)):
            await _reply(update, ask_text(lang, pending))


async def cmd_cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE | None):
    if not await _guard(update):
        return
    user = await _io(_user, update)
    lang = _lang(update, user)
    if not user.get("pending"):
        await _reply(update, t(lang, "nothing_to_cancel"))
        return
    if await _stored(update, lang, lambda: _store().set_pending(_uid(update), None)):
        await _reply(update, t(lang, "cancelled"))


# ---------- data commands ----------

def _today_data(update: Update):
    bq = _bq(update)
    return bq.get_last_activity(), bq.get_health_last_days(4), bq.get_load_last_days(1), bq.get_source_status()


async def cmd_today(update: Update, ctx: ContextTypes.DEFAULT_TYPE | None):
    if not await _guard(update):
        return
    lang = await _alang(update)
    act, health, loads, status = await _io(_today_data, update)
    load = loads[-1] if loads else None
    parts = [f"*{t(lang, 'today.last_workout')}*",
             _fmt_activity(act, lang) if act else t(lang, "today.no_activities")]
    if health:
        parts += ["", f"*{t(lang, 'today.health')}*", _fmt_health(health[-1], lang)]
    if load:
        parts += ["", f"*{t(lang, 'today.load')}*", _md(_fmt_load(load, lang))]
    parts += ["", _md(_estimates(health, load, lang))]
    parts += stale_warnings(status, _enabled_sources(), lang)
    await _reply(update, "\n".join(parts), parse_mode="Markdown")


async def cmd_week(update: Update, ctx: ContextTypes.DEFAULT_TYPE | None):
    if not await _guard(update):
        return
    lang = await _alang(update)
    acts = await _io(lambda: _bq(update).get_week_activities())
    if not acts:
        await _reply(update, t(lang, "week.empty"))
        return
    total_dist = sum((a.get("distance_meters") or 0) for a in acts) / 1000
    total_dur = sum((a.get("duration_seconds") or 0) for a in acts) // 60
    parts = ["*" + t(lang, "week.header", n=len(acts), km=round(total_dist, 1), min=total_dur) + "*\n"]
    parts += [_fmt_activity(a, lang) for a in acts]
    await _reply(update, "\n\n".join(parts), parse_mode="Markdown")


async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE | None):
    if not await _guard(update):
        return
    lang = await _alang(update)
    rows = await _io(lambda: _bq(update).get_source_status())
    enabled = _enabled_sources()
    lines = [t(lang, "status.sources", list=", ".join(enabled) or "—")]
    for r in rows:
        mark = "✅" if r.get("last_status") == "ok" else "❌"
        when = f"{r['last_success_at']:%d.%m %H:%M} UTC" if r.get("last_success_at") else t(lang, "status.never")
        lines.append(t(lang, "status.line", mark=mark, source=r["source"], when=when,
                       cursor=r.get("cursor_date") or "—"))
    lines += stale_warnings(rows, enabled, lang)
    await _reply(update, "\n".join(lines))


# ---------- AI ----------

async def _send_ai_text(update: Update, markdown: str, markup=None):
    """Claude's Markdown -> Telegram HTML, split to fit; plain text if Telegram rejects the markup.
    `markup` (buttons) goes under the last part."""
    chunks = split_message(md_to_html(markdown))
    for i, chunk in enumerate(chunks):
        kw = {"reply_markup": markup} if markup is not None and i == len(chunks) - 1 else {}
        try:
            await _reply(update, chunk, parse_mode="HTML", **kw)
        except BadRequest as e:
            logger.warning("HTML rejected by Telegram (%s), sending this part as plain text", e)
            await _reply(update, html_to_plain(chunk), **kw)


def _more_button(lang: str, entry_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton(t(lang, "btn.more"), callback_data=f"more:{entry_id}")]])


def _record_usage(update: Update, answer) -> None:
    try:
        _store().add_usage(_uid(update), _month(), answer.input_tokens, answer.output_tokens, answer.cost_usd)
        guard = _store().get_user(_uid(update)).get("guard") or {}
        _store().update_guard(_uid(update), record_spend(guard, datetime.now(timezone.utc), answer.cost_usd))
    except Exception as e:  # noqa: BLE001 - spending stats must never block the answer
        logger.error("usage write failed: %s", e)


def _athlete(update: Update, user: dict) -> str:
    """The athlete note for the model: profile, goal, zones (reads Firestore and BigQuery)."""
    try:
        goal = _store().get_goal(_uid(update))
    except Exception as e:  # noqa: BLE001
        logger.error("goal read failed: %s", e)
        goal = None
    profile = user.get("profile") or {}
    inputs = _athlete_inputs(update)
    lthr_then = ((inputs["lthr_then"], str(inputs["lthr_then_date"]))
                 if inputs.get("lthr_then") and inputs.get("lthr_then_date") else None)
    return athlete_note(profile, goal, date.today(),
                        zones=compute_zones(profile, inputs, inputs.get("resting_hr_7d")),
                        garmin_resting_hr=inputs.get("resting_hr_7d"), lthr_then=lthr_then)


EDIT_EVERY_S = 1.5         # Telegram allows about one edit per second in a chat
PREVIEW_CHARS = 4000       # one Telegram message; the rest comes after the answer is complete
CURSOR = " ▌"


def _preview(text: str, ask: bool) -> str:
    """What to show of a half-written answer: nothing until the scope line is complete, nothing at
    all of an off-topic answer, plain text (Markdown may be unbalanced) cut to one message."""
    if ask and "\n" not in text and len(text) < 40:
        return ""
    on_topic, body = split_scope(text) if ask else (None, text.strip())
    if on_topic is False:
        return ""
    return body[:PREVIEW_CHARS] + ("…" if len(body) > PREVIEW_CHARS else "")


async def _edit(msg, text: str, **kw) -> bool:
    try:
        await msg.edit_text(text, **kw)
        return True
    except BadRequest as e:
        if "not modified" in str(e).lower():
            return True
        logger.warning("edit rejected by Telegram (%s)", e)
        return False


async def _finish(update: Update, placeholder, markdown: str, markup=None):
    """Replace the live message with the formatted answer; the rest goes in new messages."""
    chunks = split_message(md_to_html(markdown))
    last = len(chunks) - 1
    kw0 = {"reply_markup": markup} if markup is not None and last == 0 else {}
    if not await _edit(placeholder, chunks[0], parse_mode="HTML", **kw0):
        await _edit(placeholder, html_to_plain(chunks[0]), **kw0)
    for i, chunk in enumerate(chunks[1:], 1):
        kw = {"reply_markup": markup} if markup is not None and i == last else {}
        try:
            await _reply(update, chunk, parse_mode="HTML", **kw)
        except BadRequest as e:
            logger.warning("HTML rejected by Telegram (%s), sending this part as plain text", e)
            await _reply(update, html_to_plain(chunk), **kw)


async def _ai_reply(update: Update, prompt: str, label: str, task: str, detail: bool = False,
                    cache_key: str | None = None, **fmt):
    user = await _io(_user, update)
    lang = _lang(update, user)
    mode = (user.get("settings") or {}).get("mode", DEFAULT_MODE)
    if not budget_allowed(user.get("guard") or {}, datetime.now(timezone.utc), _daily_budget()):
        await _reply(update, t(lang, "filter.budget_spent"))
        return
    athlete_job = asyncio.ensure_future(_io(_athlete, update, user))   # read alongside the data below
    context_job = asyncio.ensure_future(_io(lambda: _context(_bq(update))))
    placeholder = await _reply(update, t(lang, "ai.working", label=t(lang, f"ai.label.{label}")))
    live = hasattr(placeholder, "edit_text")          # the answer grows in this message while it is written
    ask = prompt is ASK_PROMPT
    pieces: list[str] = []
    lock = threading.Lock()

    def on_text(piece: str):
        with lock:
            pieces.append(piece)

    async def say(text: str, **kw):
        if not (live and await _edit(placeholder, text, **kw)):
            await _reply(update, text, **kw)

    try:
        athlete, context = await athlete_job, await context_job
        job = asyncio.ensure_future(_io(lambda: analyze(_llm(), prompt, context, task, lang, mode=mode,
                                                        athlete=athlete, on_text=on_text if live else None,
                                                        detail=detail, **fmt)))
        shown = ""
        while not job.done():
            await asyncio.wait({job}, timeout=EDIT_EVERY_S)
            if live and not job.done():
                with lock:
                    preview = _preview("".join(pieces), ask)
                if preview and preview != shown:
                    await _edit(placeholder, preview + CURSOR)
                    shown = preview
        answer = job.result()
        await _io(_record_usage, update, answer)
        text = answer.text
        if prompt is ASK_PROMPT:                         # second lock: the answer's own scope line
            on_topic, text = split_scope(text)
            if on_topic is False:
                logger.info("scope line: off topic, fixed text shown")
                await say(t(lang, "filter.off_topic"))
                return
            if on_topic is None:
                logger.warning("scope line missing in the answer (model=%s)", answer.model)
        if not text:
            await say(t(lang, "ai.error"))
            return
        entry_id = await _io(_remember, update, fmt["question"], text) if ask and fmt.get("question") else None
        markup = _more_button(lang, entry_id) if entry_id and not detail else None
        if live:
            await _finish(update, placeholder, text, markup)
        else:
            await _send_ai_text(update, text, markup)
        if ask and not detail and not fmt.get("dialog") and cache_key:
            await _io(_cache_answer, update, cache_key, text)
    except Refused as e:
        logger.warning("Claude refused: %s", e)
        await say(t(lang, "ai.refused"))
    except Exception as e:  # noqa: BLE001
        logger.error("Claude error: %s", e)
        await say(t(lang, "ai.error"))


async def cmd_coach(update: Update, ctx: ContextTypes.DEFAULT_TYPE | None):
    if await _guard(update):
        await _ai_reply(update, COACH_PROMPT, "coach", "analysis")


async def cmd_health(update: Update, ctx: ContextTypes.DEFAULT_TYPE | None):
    if await _guard(update):
        await _ai_reply(update, HEALTH_PROMPT, "health", "analysis")


async def cmd_nutrition(update: Update, ctx: ContextTypes.DEFAULT_TYPE | None):
    if await _guard(update):
        await _ai_reply(update, NUTRITION_PROMPT, "nutrition", "analysis")


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE | None):
    """An answer to a pending question (profile/goal input), otherwise a free-form question."""
    if not await _guard(update) or update.message is None or update.message.text is None:
        return
    text_in = update.message.text
    user = await _io(_user, update)
    pending = user.get("pending")
    if pending and not is_red_flag(text_in):   # an emergency is never taken as form input
        lang = _lang(update, user)

        def take_input():
            watch = _athlete_inputs(update) if pending.get("kind") == "profile" else None
            return handle_input(_store(), _uid(update), lang, pending, text_in, date.today(), watch)
        try:
            text, kb = await _io(take_input)
        except Exception as e:  # noqa: BLE001
            logger.error("input handling failed: %s", e)
            await _reply(update, t(lang, "storage.error"))
            return
        await _show(update, text, kb)
        return
    question = text_in[:1000]
    r = await _io(_guarded_route, update, user, text_in, datetime.now(timezone.utc))
    if await _reply_without_model(update, r, user, question):
        return
    history = await _io(_recent_dialog, update)
    # Read alongside the classifier: the answer cache key needs the data version.
    version_job = asyncio.ensure_future(_io(_data_version, update))
    # Nothing reaches the answering model unless the classifier recognized it as on topic.
    verdict = await _io(_scope_check, update, question, dialog.previous_questions(history))
    if verdict.follow_up and verdict.question != question:
        # "and yesterday?" -> "resting HR yesterday". The rewrite goes through the filter again
        # (emergencies, instruction changes); facts come from the classifier's label, not from the
        # rule intents, which would cut "and is that a lot?" down to a number.
        question = verdict.question
        r = route_message(question, r.lang, date.today())
        if r.kind == "filter" and await _reply_without_model(update, r, user, question):
            version_job.cancel()
            return
    if await _reply_without_model(update, scope_route(verdict, question, r.lang), user, question):
        version_job.cancel()
        return
    follow = verdict.follow_up and bool(history)
    version = await version_job
    key = None if follow else await _io(_answer_key, update, user, question, r.lang, version)
    cached = await _io(_cached_answer, update, key) if key else None
    logger.info("question fp=%s cache=%s", answer_cache.fingerprint(_uid(update), question),
                "skip" if key is None else ("hit" if cached else "miss"))
    if cached:
        entry_id = await _io(_remember, update, question, cached)
        await _send_ai_text(update, cached, _more_button(_lang(update, user), entry_id) if entry_id else None)
        return
    await _ai_reply(update, ASK_PROMPT, "ask", "chat", cache_key=key, question=question,
                    dialog=dialog.block(history) if follow else None)


ANSWER_CACHE_KEEP = answer_cache.KEEP
# Part of the answer cache key: a changed prompt means new answers.
PROMPT_VERSION = hashlib.sha256((ASK_PROMPT + PERSONA + FORMAT_NOTE).encode()).hexdigest()[:12]


def _data_version(update: Update) -> str:
    """Changes with every successful sync; "" when unknown (then nothing is cached)."""
    try:
        times = [r["last_success_at"] for r in _bq(update).get_source_status() if r.get("last_success_at")]
        return max(times).isoformat() if times else ""
    except Exception as e:  # noqa: BLE001
        logger.error("data version read failed: %s", e)
        return ""


def _answer_key(update: Update, user: dict, question: str, lang: str, version: str | None = None) -> str | None:
    version = _data_version(update) if version is None else version
    if not version:
        return None
    try:
        goal = _store().get_goal(_uid(update))
    except Exception:  # noqa: BLE001
        return None
    return answer_cache.key(_uid(update), question, lang=lang,
                            mode=(user.get("settings") or {}).get("mode", DEFAULT_MODE), data_version=version,
                            profile=user.get("profile") or {}, goal=goal, day=date.today().isoformat(),
                            prompt_version=PROMPT_VERSION)


def _cached_answer(update: Update, key: str) -> str | None:
    try:
        return _store().get_cached_answer(_uid(update), key)
    except Exception as e:  # noqa: BLE001
        logger.error("answer cache read failed: %s", e)
        return None


def _cache_answer(update: Update, key: str, text: str) -> None:
    try:
        _store().put_cached_answer(_uid(update), key, text)
    except Exception as e:  # noqa: BLE001
        logger.error("answer cache write failed: %s", e)


REMEMBERED_FILTERS = ("filter.red_flag", "filter.emergency", "filter.crisis")   # the next answer must know


def _recent_dialog(update: Update) -> list[dict]:
    try:
        return dialog.recent(_store().get_dialog(_uid(update)), datetime.now(timezone.utc))
    except Exception as e:  # noqa: BLE001 - the history is a convenience
        logger.error("dialog read failed: %s", e)
        return []


def _remember(update: Update, question: str, answer: str) -> str | None:
    """Save the exchange; its id is what the "More" button refers to."""
    e = dialog.entry(question, html_to_plain(answer), datetime.now(timezone.utc))
    try:
        _store().add_dialog(_uid(update), e)
        return e["id"]
    except Exception as ex:  # noqa: BLE001
        logger.error("dialog write failed: %s", ex)
        return None


async def _reply_without_model(update: Update, r: Route, user: dict, question: str | None = None) -> bool:
    if r.kind == "model":
        logger.info("route kind=model")
        return False
    try:
        answer = await _io(_fast_answer, update, r, user)
    except Exception as e:  # noqa: BLE001 - a failed fact or calculation is answered by the model
        logger.error("fast answer %s failed: %s", r.key, e)
        answer = None
    logger.info("route kind=%s key=%s answered_without_ai=%s", r.kind, r.key, answer is not None)
    if answer is None:
        return False
    text, kw = answer
    await _reply(update, text, **kw)
    if question and (r.kind == "intent" or r.key in REMEMBERED_FILTERS):
        await _io(_remember, update, question, text)
    return True


def _scope_check(update: Update, question: str, previous: list[str]) -> Verdict:
    verdict = check(_llm(), question, previous)
    for response in verdict.responses:
        _record_usage(update, response)
    if verdict.label == SCOPE_ERROR:
        try:
            guard = _store().get_user(_uid(update)).get("guard") or {}
            _store().update_guard(_uid(update), forget_message(guard))
        except Exception as e:  # noqa: BLE001
            logger.error("guard write failed: %s", e)
    return verdict


# ---------- answers without the AI model (filter and rule router) ----------

def _d(v) -> str:
    return v.strftime("%d.%m") if hasattr(v, "strftime") else str(v)


def _clock(seconds: float) -> str:
    s = int(round(seconds))
    h, rest = divmod(s, 3600)
    m, sec = divmod(rest, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _fast_metric(update: Update, lang: str, metric: str, day: date) -> str:
    if metric == "vo2max":
        load = _bq(update).get_load_last_days(1)
        v = load[-1].get("vo2max") if load else None
        return t(lang, "fast.metric.vo2max", v=f"{v:.1f}", date=_d(load[-1]["date"])) if v else t(lang, "fast.no_data")
    rows = _bq(update).get_health_last_days(4)
    row = next((h for h in rows if str(h.get("date")) == day.isoformat()), None) or {}
    when = _d(day)
    if metric == "resting_hr" and row.get("resting_hr"):
        return t(lang, "fast.metric.resting_hr", date=when, v=row["resting_hr"])
    if metric == "hrv" and row.get("hrv_last_night"):
        extra = (t(lang, "fast.metric.hrv_extra", lo=row["hrv_baseline_low"], hi=row["hrv_baseline_high"],
                   status=row.get("hrv_status") or "—") if row.get("hrv_baseline_low") else "")
        return t(lang, "fast.metric.hrv", date=when, v=row["hrv_last_night"], extra=extra)
    if metric == "sleep" and row.get("sleep_seconds"):
        extra = t(lang, "fast.metric.sleep_extra", deep=_hm(row.get("deep_sleep_seconds"), lang),
                  rem=_hm(row.get("rem_sleep_seconds"), lang), score=row.get("sleep_score") or "—")
        return t(lang, "fast.metric.sleep", date=when, v=_hm(row["sleep_seconds"], lang), extra=extra)
    if metric == "steps" and row.get("steps") is not None:
        return t(lang, "fast.metric.steps", date=when, v=row["steps"])
    if metric == "body_battery" and row.get("body_battery_high") is not None:
        return t(lang, "fast.metric.body_battery", date=when, lo=row.get("body_battery_low") or "—",
                 hi=row["body_battery_high"])
    if metric == "stress" and row.get("stress_avg") is not None:
        return t(lang, "fast.metric.stress", date=when, v=row["stress_avg"])
    return t(lang, "fast.metric.missing", date=when)


def _guarded_route(update: Update, user: dict, text: str, now: datetime) -> Route:
    """Rules first, then the per-user guard: repeats and the hourly model-call limit (2.1)."""
    r = route_message(text, _lang(update, user), now.date())
    guard = user.get("guard") or {}
    if r.key != "filter.red_flag" and is_repeat(guard, text, now):
        r = Route("filter", "filter.repeat", r.lang)
    elif r.kind == "model" and not model_allowed(guard, now, MODEL_CALLS_PER_HOUR):
        r = Route("filter", "filter.rate_limited", r.lang)
    elif r.kind == "model" and not budget_allowed(guard, now, _daily_budget()):
        r = Route("filter", "filter.budget_spent", r.lang)
    new_guard = record_message(guard, text, now)
    if r.kind == "model":
        new_guard = record_model_call(new_guard, now)
    try:
        _store().update_guard(_uid(update), new_guard)
    except Exception as e:  # noqa: BLE001 - the guard must never block an answer
        logger.error("guard write failed: %s", e)
    return r


def _period_label(lang: str, p: dict) -> str:
    if p["period"] == "month_name":
        name = t(lang, f"month.in.{p['month']}") + (f" {p['year']}" if p["year"] != date.today().year else "")
        return t(lang, "fast.period.month_name", name=name)
    return t(lang, f"fast.period.{p['period']}")


STAT_COLUMNS = {"resting_hr": "resting_hr", "hrv": "hrv_last_night", "sleep": "sleep_seconds", "steps": "steps",
                "stress": "stress_avg", "body_battery": "body_battery_high"}


def _period_stats(update: Update, lang: str, p: dict) -> str:
    """Average, lowest and highest daily value of one metric over a period."""
    metric = p["metric"]
    values = [r[STAT_COLUMNS[metric]] for r in _bq(update).get_health_period(p["start"], p["end"])
              if r.get(STAT_COLUMNS[metric]) is not None]
    name, period = t(lang, f"stat.name.{metric}"), _period_label(lang, p)
    fields = dict(name=name, period=period, start=_d(p["start"]), end=_d(p["end"]))
    if not values:
        return t(lang, "fast.stats.none", **fields)
    avg = sum(values) / len(values)
    if metric == "sleep":
        f = lambda v: _hm(v, lang)  # noqa: E731
    elif metric in ("resting_hr", "hrv"):
        f = lambda v: f"{v:.1f}".rstrip("0").rstrip(".")  # noqa: E731
    else:
        f = lambda v: f"{v:.0f}"  # noqa: E731
    return t(lang, "fast.stats", **fields, avg=f(avg), lo=f(min(values)), hi=f(max(values)),
             unit=t(lang, f"stat.unit.{metric}"), n=len(values))


RIEGEL = 1.06


def _race_prediction(update: Update, lang: str, target_km: float) -> str:
    """Riegel (T2 = T1 * (D2 / D1) ** 1.06) from the recent best effort closest to the target distance."""
    rows = [x for x in _bq(update).get_records(date.today() - timedelta(days=365))
            if x.get("span") == "recent" and x.get("secs") and (x.get("d") or 0) >= 1000]
    if not rows:
        return t(lang, "fast.race.no_data")
    src = min(rows, key=lambda x: (abs(x["d"] / 1000 - target_km), -x["d"]))
    total = src["secs"] * (target_km / (src["d"] / 1000)) ** RIEGEL
    return t(lang, "fast.race.result", target=f"{target_km:g}", time=_clock(total), pace=_clock(total / target_km),
             src=f"{src['d'] / 1000:g}", src_time=_clock(src["secs"]), src_date=_d(src.get("activity_date")))


def _fast_answer(update: Update, r: Route, user: dict) -> tuple[str, dict] | None:
    """Text and reply options for messages answered without the model; None = ask the model."""
    lang, p, html = r.lang, r.params, {"parse_mode": "HTML"}
    if r.kind == "filter":
        return t(lang, r.key), {}
    if r.key == "fact.last_workout":
        a = _bq(update).get_last_activity()
        return (_fmt_activity(a, lang), {"parse_mode": "Markdown"}) if a else (t(lang, "fast.no_data"), {})
    if r.key == "fact.records":
        return analytics.render_records(lang, _bq(update).get_records(date.today() - timedelta(days=365))), html
    if r.key == "fact.run_distance":
        per = _bq(update).get_period(p["start"], p["end"])
        fields = dict(period=_period_label(lang, p), start=_d(p["start"]), end=_d(p["end"]))
        if not per.get("runs"):
            return t(lang, "fast.run_none", **fields), html
        return t(lang, "fast.run_distance", km=f"{per['run_km']:.1f}", runs=per["runs"], **fields), html
    if r.key == "fact.metric":
        if p.get("stats"):
            return _period_stats(update, lang, p), html
        return _fast_metric(update, lang, p["metric"], p["date"]), html
    if r.key in ("calc.readiness", "fact.load"):
        bq = _bq(update)
        with ThreadPoolExecutor(2) as pool:                      # two independent queries at once
            health_q, loads_q = pool.submit(bq.get_health_last_days, 4), pool.submit(bq.get_load_last_days, 1)
        health, loads = health_q.result(), loads_q.result()
        load = loads[-1] if loads else None
    if r.key == "calc.readiness":
        return _estimates(health, load, lang), {}
    if r.key == "fact.load":
        return _fmt_load(load, lang) + "\n" + training_status(load, health, lang=lang).text(t(lang, "est.status")), {}
    if r.key == "calc.zones":
        inputs = _athlete_inputs(update)
        z = compute_zones(user.get("profile") or {}, inputs, inputs.get("resting_hr_7d"))
        if not z:
            return t(lang, "prof.zones.need"), html
        lines = [_zones_title(lang, z)] + [f"Z{i + 1}: {a}–{b}" for i, (a, b) in enumerate(z.zones)]
        return "\n".join(lines), html
    if r.key == "calc.race_predict":
        return _race_prediction(update, lang, p["distance_km"]), html
    if r.key == "calc.pace":
        km = p["distance_km"]
        km_s = f"{km:g}"
        if "total_s" in p:
            return t(lang, "fast.pace.result", km=km_s, time=_clock(p["total_s"]), pace=_clock(p["total_s"] / km)), html
        return t(lang, "fast.time.result", km=km_s, pace=_clock(p["pace_s"]), time=_clock(p["pace_s"] * km)), html
    return None


def _clears_pending(handler):
    """A command interrupts any half-finished input, so the next text isn't taken as an answer."""
    async def wrapped(update: Update, ctx: ContextTypes.DEFAULT_TYPE | None):
        def clear():
            if _user(update).get("pending"):
                _store().set_pending(_uid(update), None)
        if _allowed(update):
            try:
                await _io(clear)
            except Exception as e:  # noqa: BLE001
                logger.error("state write failed: %s", e)
        await handler(update, ctx)
    return wrapped


Handler = Callable[[Update, ContextTypes.DEFAULT_TYPE | None], Awaitable[None]]
COMMANDS: dict[str, Handler] = {
    "menu": cmd_menu, "today": cmd_today, "week": cmd_week, "coach": cmd_coach,
    "health": cmd_health, "nutrition": cmd_nutrition, "goal": cmd_goal, "profile": cmd_profile,
    "settings": cmd_settings, "status": cmd_status, "records": cmd_records, "month": cmd_month,
    "compare": cmd_compare, "charts": cmd_charts,
}
# What Telegram's command handlers call (button presses go through COMMANDS directly).
COMMAND_HANDLERS = {**{k: _clears_pending(v) for k, v in COMMANDS.items()}, "cancel": cmd_cancel}
