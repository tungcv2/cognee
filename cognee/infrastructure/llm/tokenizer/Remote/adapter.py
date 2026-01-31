import requests
from typing import List, Any
from ..tokenizer_interface import TokenizerInterface

class RemoteTokenizer(TokenizerInterface):
    """
    Implements a tokenizer that calls a remote API to count tokens.
    """

    def __init__(self, endpoint: str, model: str):
        self.endpoint = endpoint
        self.model = model

    def extract_tokens(self, text: str) -> List[Any]:
        raise NotImplementedError("RemoteTokenizer does not support token extraction.")

    def count_tokens(self, text: str) -> int:
        try:
            response = requests.post(
                self.endpoint,
                json={
                    "model": self.model,
                    "input": text
                },
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return data.get("token_count", 0)
        except Exception as e:
            # Fallback to a rough estimate or log error
            from cognee.shared.logging_utils import get_logger
            logger = get_logger("RemoteTokenizer")
            logger.error(f"Error calling remote tokenizer API: {e}")
            # Character based estimate as fallback (rough)
            return len(text) // 4

    def decode_single_token(self, token: int) -> str:
        raise NotImplementedError("RemoteTokenizer does not support token decoding.")
