from typing import Any, List, Mapping, Optional

import requests
import torch
from app.config import (
    CUSTOM_LLM,
    CUSTOM_LLM_API,
    CUSTOM_LLM_ENABLED,
    CUSTOM_LLM_OPTIONS,
    LLM_CHECKPOINT_ID,
    LLM_TASK,
    LLM_TEMPERATURE,
)
from langchain.llms.base import LLM
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


class CustomLLM(LLM):

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

        response = requests.post(CUSTOM_LLM_API, json=payload, verify=False, timeout=90)
        response.raise_for_status()
        return response.json()["response"]  # response from the API

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"llmUrl": self._llm_type}


def custom_llm() -> CustomLLM:
    """Return the custom LLM."""
    return CustomLLM()


def hg_llm() -> HuggingFacePipeline:
    """Return the HuggingFace LLM."""
    tokenizer = AutoTokenizer.from_pretrained(LLM_CHECKPOINT_ID, return_tensors="pt")

    if "t5" in LLM_CHECKPOINT_ID:
        model = AutoModelForSeq2SeqLM.from_pretrained(LLM_CHECKPOINT_ID)
    else:
        model = LLM_CHECKPOINT_ID

    pipe = pipeline(
        task=LLM_TASK,
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        do_sample=True,
        temperature=LLM_TEMPERATURE,
    )

    return HuggingFacePipeline(pipeline=pipe)


llm = custom_llm() if CUSTOM_LLM_ENABLED else hg_llm()
