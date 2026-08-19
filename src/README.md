# GPQA with user knowledge

`run_gpqa_with_knowledge.py` は、指定したsystem promptテンプレートへ知識テキストの
**全文**を埋め込み、Ollama Cloud APIでGPQAを実行・採点します。
`KIOKU-AI/GPQA` にある上流GPQAの既存ファイルや依存関係は変更しません。
実行結果は、既定でコマンドを実行したカレントディレクトリ直下の `run/` に保存されます。
`run/` は `.gitignore` の対象です。

system promptテンプレートは `--system-prompt-file` で指定します。未指定の場合、
system promptは空文字列になります。従来のsystem promptは
`src/sample/system-prompt.txt` に保存してあり、ファイル内の `{{knowledge}}` が知識全文へ
置換されます。

## 最短の実行方法

`KIOKU-AI` の直下に `.env` を作成します。

```dotenv
API_KEY=...
```

system promptを設定する場合:

```bash
python src/run_gpqa_with_knowledge.py src/sample/knowledge.txt \
  --system-prompt-file src/sample/system-prompt.txt \
  --output-dir run/main-with-system-prompt
```

system promptを空にする場合:

```bash
python src/run_gpqa_with_knowledge.py src/sample/knowledge.txt \
  --output-dir run/main-empty-system-prompt
```

デフォルトでは、`GPQA/dataset.zip` 内の `gpqa_main.csv` を直接読みます。
ZIPの公開パスワードは上流README記載値を使用します。別のパスワードを使う場合は
`GPQA_DATASET_PASSWORD` を設定してください。

APIを呼ばず、入力とprompt構築を検証するには次を実行します。

```bash
python src/run_gpqa_with_knowledge.py src/sample/knowledge.txt \
  --system-prompt-file src/sample/system-prompt.txt \
  --dry-run --max-examples 2
```

## 検証済みサンプル

`src/sample/knowledge.txt` を使い、Ollama Cloudの `gpt-oss:120b` で
GPQA Diamondの先頭5問を実行した結果を保存しています。

```bash
python src/run_gpqa_with_knowledge.py src/sample/knowledge.txt \
  --system-prompt-file src/sample/system-prompt.txt \
  --split diamond \
  --max-examples 5 \
  --model gpt-oss:120b \
  --base-url https://ollama.com \
  --output-dir run/sample \
  --overwrite-output \
  --retries 0
```

- `run/sample/results.jsonl`: 5問の問題別結果
- `run/sample/summary.json`: 5問中4問正解、accuracy 0.8、JSON parse 5/5、API error 0
- サンプル用知識には追加の分野固有知識を含めていません。実利用時は任意の知識ファイルへ置き換えてください。

## GPQA Diamond全198問の実行結果

同じサンプル知識、seed 0、`gpt-oss:120b`、Ollama Cloud直接APIの条件で
Diamond全198問を逐次実行した結果を次に保存しています。

- `run/diamond-198-gpt-oss-120b-20260819/results.jsonl`
- `run/diamond-198-gpt-oss-120b-20260819/summary.json`
- 正解138/198、accuracy 0.696969696969697
- JSON parse 198/198、API error 0、除外0
- prompt evaluation 103,051 tokens、completion evaluation 283,779 tokens
- 実行時間は約14分27秒

## よく使う指定

```bash
python src/run_gpqa_with_knowledge.py src/sample/knowledge.txt \
  --system-prompt-file src/sample/system-prompt.txt \
  --split diamond \
  --model gpt-oss:120b \
  --base-url https://ollama.com \
  --max-examples 10 \
  --seed 0 \
  --output-dir run/diamond-10
```

- `--data-file`: 公式形式のCSV、または公式 `dataset.zip` を指定します。
- `--system-prompt-file`: `{{knowledge}}` を含むUTF-8のsystem promptテンプレートです。
  未指定時は空のsystem promptを送ります。
- `--split`: `main`、`diamond`、`experts`、`extended` から選びます。
- `--model`: 既定値はOllama Cloud APIで利用可能な `gpt-oss:120b` です。
- `--base-url`: 既定値は `https://ollama.com` です。
- `--env-file`: 既定値はカレント直下の `.env` です。
- `--api-key-env`: `.env`からAPIキーを読むキー名で、既定値は `API_KEY` です。
- `--temperature`: ローカルOllamaまたはOpenAI互換API用です。Ollama Cloud直結では
  公式の最小payloadに合わせて省略します。
- `--output-dir`: 結果の保存先です。省略時はカレント直下の `run/` 以下に
  timestamp付きの新規ディレクトリを作成します。

`GPQA_DATA_FILE` でもデータファイルを指定できます。`--api-style openai` を指定すると、
OpenAI互換 `/v1/chat/completions` 形式も利用できます。

## 出力と採点

- `results.jsonl`: 問題、シャッフル後の選択肢、正解、モデル回答、正誤、usage、エラー
- `summary.json`: accuracy、parse件数、API error件数、usage合計、入力ハッシュ

選択肢は上流 `baselines/utils.py` と同様にseed付きでシャッフルします。
モデルには `{"answer":"A"}` だけを返すよう求めます。JSONがその単一キー形式であり、
値がA〜Dである場合だけ有効回答として正解indexと比較します。APIエラーや不正JSONは
正解数に含めません。Ollama CloudはJSON Schema強制をサポートしないため、promptと
クライアント側検証で形式を保証します。

## 前提と注意

- Python 3.9以上。追加パッケージは不要です。
- 知識ファイルとCSVはUTF-8を前提とします。
- 知識全文を**各問題ごと**に送るため、コンテキスト上限、API利用量、費用を事前に確認してください。
- 実行結果を比較する場合は、モデルの固定version、seed、temperature、知識ファイルの
  SHA-256を揃えてください。
- Ollama CloudへはBearer認証で接続します。APIキー値は出力ファイルへ保存しません。
