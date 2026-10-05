from typing import Any

import pytest

from tokenizer_tools import is_openai_gpt2, load_tokenizer


def tokens(tokenizer, text, **kwargs):
    return tokenizer.convert_ids_to_tokens(tokenizer(text, **kwargs)["input_ids"])


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("gpt2", True),
        ("openai-community/gpt2", True),
        ("openai-community/gpt2-xl", True),
        ("rinna/japanese-gpt2-small", False),  # "gpt2" in the name only
        ("gpt2-japanese", False),
        ("distilbert/distilgpt2", False),
    ],
)
def test_is_openai_gpt2(name, expected):
    assert is_openai_gpt2(name) is expected


network = pytest.mark.network


@network
@pytest.mark.parametrize("name", ["gpt2", "openai-community/gpt2"])
def test_gpt2_gets_bos(name):
    tokenizer = load_tokenizer(name)
    assert tokens(tokenizer, "Tokyo is") == ["<|endoftext|>", "Tok", "yo", "Ġis"]
    assert tokens(tokenizer, "Tokyo is", add_special_tokens=False) == [
        "Tok",
        "yo",
        "Ġis",
    ]


@network
def test_rinna_gpt2_small_is_lowercased_without_special_tokens():
    tokenizer = load_tokenizer("rinna/japanese-gpt2-small")
    expected = ["▁", "nasa", "と", "tokyo", "大学の", "abc", "研究"]
    assert tokens(tokenizer, "NASAとTokyo大学のABC研究") == expected
    assert tokens(tokenizer, "NASAとTokyo大学のABC研究", add_special_tokens=False) == (
        expected
    )
    batch = tokenizer(["NASA", "日本"])["input_ids"]
    assert [tokenizer.convert_ids_to_tokens(ids) for ids in batch] == [
        ["▁", "nasa"],
        ["▁日本"],
    ]


@network
def test_rinna_gpt_1b_keeps_case_without_special_tokens():
    tokenizer = load_tokenizer("rinna/japanese-gpt-1b")
    assert tokens(tokenizer, "東京とNASA") == ["▁東京", "と", "NA", "SA"]


@network
def test_t5_keeps_its_eos():
    # an encoder-decoder: </s> is what T5 expects
    tokenizer = load_tokenizer("google-t5/t5-small")
    assert tokens(tokenizer, "Hello")[-1] == "</s>"


@network
def test_other_tokenizers_are_unchanged():
    from transformers import AutoTokenizer

    name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # adds <s> itself
    text = "Tokyo is the capital of Japan."
    plain: Any = AutoTokenizer.from_pretrained(name)
    assert load_tokenizer(name)(text)["input_ids"] == plain(text)["input_ids"]
