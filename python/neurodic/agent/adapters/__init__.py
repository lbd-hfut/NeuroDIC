"""Solver-specific, read-only workflow descriptions for inspection."""

from . import ndef, pin, pin_multi, pin_stereo

ADAPTERS = {"pin": pin, "pin_stereo": pin_stereo, "pin_multi": pin_multi, "ndef": ndef}
