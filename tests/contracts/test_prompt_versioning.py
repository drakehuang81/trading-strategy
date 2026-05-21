import hashlib
from pathlib import Path

from models.llm.gemma_context import GemmaContextProvider, PROMPT_PATH


def test_gemma_prompt_version_matches_file_hash():
    expected = hashlib.sha256(Path(PROMPT_PATH).read_bytes()).hexdigest()
    provider = GemmaContextProvider(client=None)      # type: ignore[arg-type]
    assert provider.prompt_version == expected
