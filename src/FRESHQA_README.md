# FreshQA with user knowledge

`run_freshqa_with_knowledge.py` は、指定した `knowledge.txt` の全文をsystem
promptへ入れ、FreshQAの自由記述問題へOllama `/api/chat` またはOpenAI互換APIで
回答します。既存の
`run_gpqa_with_knowledge.py` とは完全に独立したCLIであり、`FreshQA/` と `GPQA/`
のファイルは変更しません。

## 準備

FreshQAリポジトリのREADMEにある公式スプレッドシートをCSV形式でダウンロードし、
`src/data/freshqa.csv` として保存します。CLIは、公式FreshEval notebookと
同じ `question`、`answer_0`〜`answer_9` 列を読み取ります。CSVの先頭にタイトルや
説明行があっても、ヘッダー行を自動検出します。

既定では、既存GPQA用CLIと同じOllama Cloudの `/api/chat` と
`gpt-oss:120b` を回答に使い、judgeには `qwen3.8:27b` を使います。
judgeの既定URLはローカルOllamaの `http://localhost:11434` です。
ルートの `.env` にAPIキーを設定します。

```dotenv
API_KEY=...
```

`OLLAMA_MODEL`、`OLLAMA_BASE_URL`、または対応する `FRESHQA_*` でも上書きできます。
judgeは `OLLAMA_JUDGE_MODEL` または `FRESHQA_JUDGE_MODEL` でも上書きできます。
judge URLは `OLLAMA_JUDGE_BASE_URL` または `FRESHQA_JUDGE_BASE_URL` でも指定できます。
APIキー値は結果へ保存しません。

## 実行

最小構成:

```bash
python src/run_freshqa_with_knowledge.py knowledge.txt
```

データ、モデル、APIを明示する例:

```bash
python src/run_freshqa_with_knowledge.py knowledge.txt \
  --data-file src/data/freshqa.csv \
  --system-prompt-file src/sample/system-prompt.txt \
  --api-style ollama \
  --model gpt-oss:120b \
  --base-url https://ollama.com \
  --evaluator strict \
  --judge-api-style ollama \
  --judge-model qwen3.8:27b \
  --judge-base-url http://localhost:11434 \
  --max-examples 10 \
  --output-dir run/freshqa-10
```

APIを呼ばずにCSV、知識全文、prompt構築、対象件数を検証する例:

```bash
python src/run_freshqa_with_knowledge.py knowledge.txt \
  --data-file src/data/freshqa.csv \
  --dry-run \
  --max-examples 2
```

環境変数 `FRESHQA_DATA_FILE` でもCSVを指定できます。`--split TEST` はCSVの
`split` 列を大文字小文字を区別せず絞り込みます。

## system prompt

`--system-prompt-file` には、`{{knowledge}}` を含むUTF-8のテンプレートを指定できます。
この部分が知識全文へ置換されます。指定例として既存の
`src/sample/system-prompt.txt` を利用できます。

```bash
python src/run_freshqa_with_knowledge.py knowledge.txt \
  --data-file src/data/freshqa.csv \
  --system-prompt-file src/sample/system-prompt.txt \
  --dry-run
```

未指定時は、次の趣旨を明示したFreshQA組み込みsystem promptへ知識全文を挿入します。

> このテキストファイルは質問回答のために利用可能な知識です。利用可能な知識を
> 抽出して利用し、ユーザーの質問に回答してください。

問題に誤った前提がある場合は明示的に訂正するよう、各FreshQA質問のuser promptでも
指示します。

## 評価と出力

既定の `--evaluator strict` は、回答生成後に同じAPIをFreshEval strict
基準のjudgeとして呼び出します。`--evaluator relaxed` では、主回答に影響しない副次的な
古い情報や不正確さを許容するFreshEval relaxed基準を使います。別APIを使う場合は
`--judge-api-style`、`--judge-base-url`、`--judge-model`、`--judge-api-key-env` を指定します。

たとえば回答とjudgeを別のモデル/APIへ明示的に振り分けられます。

```bash
python src/run_freshqa_with_knowledge.py knowledge.txt \
  --model gpt-oss:120b \
  --base-url https://ollama.com \
  --judge-model qwen3.8:27b \
  --judge-base-url http://localhost:11434
```

Ollama Libraryにモデルタグが存在しても、Ollama Cloud APIで同じタグが利用できるとは
限りません。`qwen3.8:27b` をローカルOllamaへpullして利用する場合は、上記のように
`--judge-base-url http://localhost:11434` を指定します。別のホストやgatewayで配信する
場合も、そのbase URLを同じオプションへ指定してください。

OpenAI互換 `/chat/completions` を使う場合:

```bash
python src/run_freshqa_with_knowledge.py knowledge.txt \
  --data-file src/data/freshqa.csv \
  --api-style openai \
  --model openai-compatible-model \
  --base-url https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY
```

このCLIは公式notebookの評価基準とCSV列に合わせていますが、廃止済みモデルを前提とする
公式few-shot promptをそのまま複製するものではありません。厳密な論文再現では、公式
notebookまたは人手評価も併用してください。

結果は既定で `run/freshqa-YYYYMMDD-HHMMSS/` に保存します。

- `results.jsonl`: 問題、参照回答、モデル回答、TRUE/FALSE判定、説明、usage、エラー
- `summary.json`: 全体accuracy、APIエラー、usage合計、入力SHA-256、カテゴリ別集計

`--evaluator none` は回答だけを保存し、正誤評価とaccuracyを生成しません。
APIエラーや解析不能なjudge応答は `unscored` として集計します。

## 前提と注意

- Python 3.9以上。追加パッケージは不要です。
- 知識ファイルとFreshQA CSVはUTF-8を前提とします。
- 知識全文を各問題の回答リクエストへ送ります。strict/relaxed評価では、通常は1問につき
  回答とjudgeの2 API呼び出しになるため、コンテキスト上限、費用、送信データを事前に
  確認してください。
- FreshQAは更新型ベンチマークです。再現可能性のため、CSVと知識ファイルのSHA-256、
  モデルversion、評価モードを結果に保存します。
