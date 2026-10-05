# tokenizer-tools

Hugging Face のトークナイザを、既知の問題を直して読み込むライブラリ。corpus-tools（頻度を数える）と transformer-feature-extractor（モデルを動かす）が同じトークナイザを使うために、両方から使う。

## インストール

```toml
# 使う側の pyproject.toml
dependencies = ["tokenizer-tools"]

[tool.uv.sources]
tokenizer-tools = { git = "https://github.com/gokamoda/tokenizer-tools.git", rev = "<commit>" }
```

開発するときは `make install`。

## 使い方

```python
from tokenizer_tools import load_tokenizer

tokenizer = load_tokenizer("rinna/japanese-gpt2-small")
tokenizer("NASAとTokyo")["input_ids"]  # 小文字にしてから: ▁ nasa と tokyo
```

`load_tokenizer(name, **kwargs)` は `AutoTokenizer.from_pretrained(name, **kwargs)` と同じものを返す。ただし、次の問題だけを読み込むときに直す。それ以外の振る舞い（特殊トークン、正規化）は、そのトークナイザのまま。

| 対象 | 問題 | 直し方 |
|---|---|---|
| OpenAI の GPT-2（`gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`。`openai-community/` は付いていてもよい） | トークナイザは BOS を付けないが、OpenAI のコードはモデルに `<\|endoftext\|>` を先に与える | 先頭に `<\|endoftext\|>` を付ける（`add_bos_token`） |
| T5Tokenizer を使う因果言語モデル（rinna の日本語 GPT） | T5Tokenizer は末尾に `</s>` を付ける（エンコーダ・デコーダの T5 の作法）が、モデルカードはテキストだけを与えている | `</s>` を付けない（`add_eos_token`）。エンコーダ・デコーダ（T5 など）はそのまま |
| トークナイザの設定が `do_lower_case: true`（`rinna/japanese-gpt2-small`, `-medium`） | 語彙に英大文字がないが、transformers 5 はこの設定で小文字にしないので、大文字が未知のトークンになる | トークナイズの前に小文字にする（トークナイザの正規化の先頭に足す） |

特殊トークンの修正は、`add_special_tokens=True`（既定）で呼んだときに効く。`add_special_tokens=False` ではテキストのトークンだけになる（小文字化は効く）。

## 開発

- `make test`: 単体テスト。`make test-network`: Hub からトークナイザを取得して確かめるテスト。
- `make ruff`, `make ty`
