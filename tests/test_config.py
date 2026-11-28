"""Bot settings are checked once at startup; a bad value stops the start with a clear message that
names the variable but never shows its value (some are secrets). Written before the code."""

import pytest

from secondwind.apps.bot.config import BotConfig, ConfigError

GOOD = {"TELEGRAM_BOT_TOKEN": "123456:secret-token-value", "TELEGRAM_WEBHOOK_SECRET": "hook-secret-value",
        "GCP_PROJECT": "p", "BQ_DATASET": "secondwind", "TELEGRAM_CHAT_ID": "42", "OWNER_USER_ID": "u-1"}


def test_defaults():
    c = BotConfig.from_env(GOOD)
    assert c.tenancy == "solo" and c.model_calls_per_hour == 20 and c.bot_language is None
    assert c.data_sources == [] and c.telegram_bot_token.get_secret_value() == GOOD["TELEGRAM_BOT_TOKEN"]
    assert "secret-token-value" not in repr(c) and "hook-secret-value" not in str(c)


def test_values_are_parsed():
    c = BotConfig.from_env({**GOOD, "MODEL_CALLS_PER_HOUR": " 30 ", "BOT_LANGUAGE": "ru", "DATA_SOURCES": "garmin, ",
                            "TENANCY": "solo"})
    assert c.model_calls_per_hour == 30 and c.bot_language == "ru" and c.data_sources == ["garmin"]


@pytest.mark.parametrize("name,value", [("TENANCY", "Solo"), ("MODEL_CALLS_PER_HOUR", "2O"),
                                        ("MODEL_CALLS_PER_HOUR", "0"), ("BOT_LANGUAGE", "de"),
                                        ("TELEGRAM_CHAT_ID", "my chat"), ("TELEGRAM_BOT_TOKEN", "")])
def test_a_bad_value_names_the_variable_only(name, value):
    with pytest.raises(ConfigError) as e:
        BotConfig.from_env({**GOOD, name: value})
    assert name in str(e.value)
    for secret in ("secret-token-value", "hook-secret-value", value or "\0"):
        assert secret not in str(e.value) or secret in (name, "0")


def test_missing_required_settings_are_listed_together():
    with pytest.raises(ConfigError) as e:
        BotConfig.from_env({"TELEGRAM_BOT_TOKEN": "1:x"})
    assert "GCP_PROJECT" in str(e.value) and "BQ_DATASET" in str(e.value)
