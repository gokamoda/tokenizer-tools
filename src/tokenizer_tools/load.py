"""Load a Hugging Face tokenizer, with fixes for known problems.

A tokenizer keeps its own behavior (special tokens, normalization), except
for these problems, fixed on loading:

- OpenAI's GPT-2 (gpt2, gpt2-medium, gpt2-large, gpt2-xl, with or without
  "openai-community/"): its tokenizer adds no BOS, but the model is given
  <|endoftext|> first in OpenAI's code. It is added (add_bos_token).
- A causal language model with T5Tokenizer (rinna's Japanese GPT models):
  T5Tokenizer appends </s>, as T5 (an encoder-decoder) expects, but the
  model cards give the text alone. It is not appended (add_eos_token).
- do_lower_case in the tokenizer config (rinna/japanese-gpt2-small, -medium):
  their vocabulary has no upper-case letters, and transformers 5 no longer
  lowercases for this setting, so upper-case letters became unknown tokens.
  The text is lowercased first (a normalizer of the tokenizer).

The special tokens are those of a call with add_special_tokens=True (the
default); add_special_tokens=False gives the text alone, lowercased if so.
"""

from typing import Any

OPENAI_GPT2 = ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")


def is_openai_gpt2(name_or_path: str) -> bool:
    return name_or_path.removeprefix("openai-community/") in OPENAI_GPT2


def load_tokenizer(name_or_path: str, **kwargs: Any) -> Any:
    """AutoTokenizer.from_pretrained(name_or_path, **kwargs), with the fixes."""
    from transformers import AutoTokenizer

    tokenizer: Any = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if is_openai_gpt2(name_or_path):
        tokenizer.add_bos_token = True
    if type(tokenizer).__name__ == "T5Tokenizer" and _is_causal(name_or_path):
        tokenizer.add_eos_token = False
    if tokenizer.init_kwargs.get("do_lower_case"):
        _lowercase_first(tokenizer)
    return tokenizer


def _is_causal(name_or_path: str) -> bool:
    """Whether the model is not an encoder-decoder (False if unknown)."""
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(name_or_path)
    except (OSError, ValueError):  # no config.json, e.g. a tokenizer alone
        return False
    return not config.is_encoder_decoder


def _lowercase_first(tokenizer: Any) -> None:
    from tokenizers import normalizers

    backend = tokenizer.backend_tokenizer
    if backend.normalizer is None:
        backend.normalizer = normalizers.Lowercase()
    else:
        backend.normalizer = normalizers.Sequence(
            [normalizers.Lowercase(), backend.normalizer]
        )
