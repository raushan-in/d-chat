from typing import Any, List, Mapping, Optional

import requests
from config import CUSTOM_LLM, CUSTOM_LLM_API, CUSTOM_LLM_OPTIONS
from langchain.llms.base import LLM


class LlamaLLM(LLM):

    @property
    def _llm_type(self) -> str:
        return CUSTOM_LLM

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        payload = {
            "model": CUSTOM_LLM,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "options": CUSTOM_LLM_OPTIONS,
        }

        headers = {"Content-Type": "application/json"}

        response = requests.post(
            CUSTOM_LLM_API, json=payload, headers=headers, verify=False
        )
        response.raise_for_status()
        return response.json()["response"]  # get the response from the API

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"llmUrl": self._llm_type}


def custom_llm() -> LlamaLLM:
    """Return the custom LLM."""
    return LlamaLLM()
