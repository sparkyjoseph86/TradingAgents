import re
from datetime import date, datetime, timedelta
from typing import Annotated

import pandas as pd

SavePathType = Annotated[str, "File path to save data. If None, data is not saved."]

# Tickers can contain letters, digits, dot, dash, underscore, caret
# (index symbols like ^GSPC), equals (futures like GC=F), and plus
# (forex/CFD symbols like XAUUSD+). None of these enable directory
# traversal, so the value never escapes a containing directory when
# interpolated into a path. Anything else is rejected.
_TICKER_PATH_RE = re.compile(r"^[A-Za-z0-9._\-\^=+]+$")


def safe_ticker_component(value: str, *, max_len: int = 32) -> str:
    """Validate ``value`` is safe to interpolate into a filesystem path.

    Tickers come from user CLI input or from LLM tool calls, both of which
    can be influenced by attacker-controlled content (e.g. prompt injection
    embedded in fetched news). Without validation, a value like
    ``"../../../etc/foo"`` flows into ``os.path.join`` / ``Path /`` and
    escapes the configured cache, checkpoint, or results directory.

    Returns ``value`` unchanged when it matches the allowed pattern; raises
    ``ValueError`` otherwise.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"ticker must be a non-empty string, got {value!r}")
    if len(value) > max_len:
        raise ValueError(f"ticker exceeds {max_len} chars: {value!r}")
    if not _TICKER_PATH_RE.fullmatch(value):
        raise ValueError(
            f"ticker contains characters not allowed in a filesystem path: {value!r}"
        )
    # The regex above allows '.', so values like '.', '..', '...' would pass,
    # and as a path component they traverse the parent directory. Reject any
    # value that's only dots.
    if set(value) == {"."}:
        raise ValueError(f"ticker cannot consist solely of dots: {value!r}")
    return value


# Matches this codebase's prompt section delimiters, e.g. <start_of_reddit>
# or <end_of_stocktwits>. Fetched third-party content (news, Reddit posts,
# StockTwits messages) is interpolated between these tags in agent prompts
# (see sentiment_analyst.py). A post whose body contains the literal tag
# text could visually "close" the untrusted block early, letting attacker
# text that follows blend into the trusted prompt. Case-insensitive since
# the model reads this as plain text, not a case-sensitive parser.
_PROMPT_DELIMITER_TAG_RE = re.compile(r"<(/?)(start|end)_of_[a-z_]+>", re.IGNORECASE)


def defang_prompt_delimiters(text: str) -> str:
    """Neutralize literal occurrences of this codebase's prompt delimiter tags.

    Swaps the angle brackets for square brackets (``<end_of_reddit>`` ->
    ``[end_of_reddit]``), which breaks the exact token the prompt-assembly
    code matches on while leaving the text otherwise readable. This is
    defense-in-depth for delimiter break-out only — it does not make
    instructions embedded elsewhere in untrusted content safe to follow,
    since the content is still handed to an LLM reading plain text. Pair
    with an explicit "treat this as data, not instructions" framing in the
    prompt itself.
    """
    if not text:
        return text
    return _PROMPT_DELIMITER_TAG_RE.sub(
        lambda m: m.group(0).replace("<", "[").replace(">", "]"), text
    )


def save_output(data: pd.DataFrame, tag: str, save_path: SavePathType = None) -> None:
    if save_path:
        data.to_csv(save_path, encoding="utf-8")
        print(f"{tag} saved to {save_path}")


def get_current_date():
    return date.today().strftime("%Y-%m-%d")


def decorate_all_methods(decorator):
    def class_decorator(cls):
        for attr_name, attr_value in cls.__dict__.items():
            if callable(attr_value):
                setattr(cls, attr_name, decorator(attr_value))
        return cls

    return class_decorator


def get_next_weekday(date):

    if not isinstance(date, datetime):
        date = datetime.strptime(date, "%Y-%m-%d")

    if date.weekday() >= 5:
        days_to_add = 7 - date.weekday()
        next_weekday = date + timedelta(days=days_to_add)
        return next_weekday
    else:
        return date
