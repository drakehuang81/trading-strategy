import pytest
from unittest.mock import AsyncMock

from models.base import LLMContextFlags
from models.llm.gemma_context import GemmaContextProvider


@pytest.mark.asyncio
async def test_returns_flags(monkeypatch):
    fake_client = AsyncMock()
    fake_client.complete = AsyncMock(return_value=LLMContextFlags(
        context_veto=False, veto_reason=None, structural_flags=["trend"],
    ))
    provider = GemmaContextProvider(client=fake_client)
    out = await provider.flags({"smc": {}})
    assert out.structural_flags == ["trend"]
