"""Shared setup for the modules' `python -m ...` command-line entry points."""
import sys


def use_utf8_output() -> None:
    """Make print() safe for any text on Windows consoles.

    Windows terminals default to a legacy code page (cp1252), so printing LLM
    output with characters like a non-breaking hyphen or curly quotes raised
    UnicodeEncodeError and crashed the CLI after the real work had finished.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
