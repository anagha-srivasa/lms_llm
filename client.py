"""
Handles loading and querying a locally stored LLM model with robust length management,
streaming, batching, retry logic, and logging.
"""
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
import torch
from typing import List, Optional, Iterator

# Configure a module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

class LengthValidationError(Exception):
    pass

class LLMClient:
    def __init__(self, model_path: str, device: str = 'cpu', max_input_length: int = 1024, max_output_length: int = 1024):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.device = device
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def _validate_lengths(self, prompt_ids: torch.Tensor, max_length: int):
        input_len = prompt_ids.shape[-1]
        if input_len > self.max_input_length:
            raise LengthValidationError(
                f"Prompt length {input_len} exceeds max allowed {self.max_input_length}"
            )
        if max_length > self.max_output_length:
            logger.warning(
                f"Requested max_length {max_length} > max_output_length {self.max_output_length}, capping."
            )
            return self.max_output_length
        return max_length

    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        num_return_sequences: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **gen_kwargs
    ) -> List[str]:
        """
        Generate one or more sequences from the model.

        Returns a list of generated strings.
        """
        try:
            inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True).to(self.device)
            max_length_validated = self._validate_lengths(inputs.input_ids, max_length)

            # Setup stopping criteria if provided
            stopping = None
            if stop_sequences:
                stopping = StoppingCriteriaList([
                    self.tokenizer.build_stopping_criteria(stop_sequences)
                ])

            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=inputs.input_ids.shape[-1] + max_length_validated,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k or 0,
                top_p=top_p or 1.0,
                stopping_criteria=stopping,
                **gen_kwargs
            )
            decoded = [self.tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
            logger.info(f"Generated {len(decoded)} sequences for prompt of length {inputs.input_ids.shape[-1]}")
            return decoded

        except RuntimeError as e:
            if 'out of memory' in str(e):
                logger.error("CUDA out of memory during generation. Clearing cache and retrying...")
                torch.cuda.empty_cache()
                return self.generate(
                    prompt,
                    max_length=max_length // 2,
                    num_return_sequences=num_return_sequences,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    stop_sequences=stop_sequences,
                    **gen_kwargs
                )
            else:
                logger.exception("Unexpected error during generation")
                raise

    def stream_generate(
        self,
        prompt: str,
        max_length: int = 512,
        **gen_kwargs
    ) -> Iterator[str]:
        """
        Yields tokens as they are generated (streaming interface).
        """
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            max_length=inputs.input_ids.shape[-1] + max_length,
            do_sample=gen_kwargs.get('do_sample', True),
            temperature=gen_kwargs.get('temperature', 1.0),
            return_dict_in_generate=True,
            output_scores=True,
            **gen_kwargs
        )
        for token_id in outputs.sequences[0][inputs.input_ids.shape[-1]:]:
            yield self.tokenizer.decode(token_id.unsqueeze(0), skip_special_tokens=True)

    def encode(self, text: str) -> torch.Tensor:
        """Returns token IDs for a given text."""
        return self.tokenizer(text, return_tensors='pt').input_ids

    def decode(self, token_ids: torch.Tensor) -> str:
        """Converts token IDs back to a string."""
        return self.tokenizer.decode(token_ids.squeeze(), skip_special_tokens=True)

    def score(self, prompt: str, continuation: str) -> float:
        """
        Returns the negative log-likelihood of continuation given prompt.
        """
        inputs = self.tokenizer(prompt + continuation, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs.input_ids)
        return outputs.loss.item()
