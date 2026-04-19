"""Contract: ChatLLM tools are read-only — spec §4.6."""
from __future__ import annotations

from interface.tools import TOOL_NAMES


WRITE_TOOL_PATTERNS = {
    "submit", "cancel", "place", "execute", "order",
    "delete", "create", "update", "modify", "halt", "resume",
}


def test_no_write_tools_in_chat_registry():
    """ChatLLM tools must never include order-producing operations."""
    for tool_name in TOOL_NAMES:
        for pattern in WRITE_TOOL_PATTERNS:
            assert pattern not in tool_name.lower(), (
                f"Tool '{tool_name}' looks like a write operation (contains '{pattern}')"
            )


def test_tool_names_are_expected_set():
    """Guard against accidental tool additions."""
    expected = {"get_positions", "get_recent_proposals", "get_pnl_summary", "get_feature_snapshot"}
    assert TOOL_NAMES == expected
