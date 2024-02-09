import numpy as np
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging
from deita.selection.scorer.base import Scorer
import torch

logger = logging.getLogger(__name__)


class RewardScorer:
    def __init__(self, model_name_or_path: str, batch_size: int = 16):
        self.is_vllm = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            device="cuda",
            model_kwargs={"torch_dtype": torch.bfloat16},
        )

        self.pipe_kwargs = {
            "return_all_scores": True,
            "function_to_apply": "none",
            "batch_size": batch_size,
        }

    def batch_infer(self, input_text_list: list, resp_text_list: list):
        input_text_list = [f"###Human: {input_text}" for input_text in input_text_list]
        resp_text_list = [f"###Asisstant: {resp_text}" for resp_text in resp_text_list]
        input_resp_list = [f"{input_text} {resp_text}" for input_text, resp_text in zip(input_text_list, resp_text_list)]
        scores = self.pipeline(input_resp_list, **self.pipe_kwargs)
        rewards = [score[0]["score"] for score in scores]

        return rewards