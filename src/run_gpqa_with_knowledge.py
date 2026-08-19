#!/usr/bin/env python3
"""Run GPQA with a user-supplied knowledge file via the Ollama Cloud API.

The implementation intentionally uses only Python's standard library so it can
run alongside the upstream GPQA checkout without changing its dependencies.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, TextIO, Tuple


REQUIRED_COLUMNS = (
    "Question",
    "Correct Answer",
    "Incorrect Answer 1",
    "Incorrect Answer 2",
    "Incorrect Answer 3",
)
CHOICE_LETTERS = ("A", "B", "C", "D")
OFFICIAL_ARCHIVE_PASSWORD = "deserted-untie-orchid"
DEFAULT_BASE_URL = "https://ollama.com"
DEFAULT_MODEL = "gpt-oss:120b"
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
KNOWLEDGE_PLACEHOLDER = "{{knowledge}}"


@dataclass(frozen=True)
class Example:
    question_id: str
    question: str
    choices: Tuple[str, str, str, str]
    correct_index: int


class ApiError(RuntimeError):
    """An API request failed, optionally with an HTTP status code."""

    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_env_file(path: Path) -> Dict[str, str]:
    """Read a small dotenv file without exposing values or adding dependencies."""
    if not path.is_file():
        raise ValueError(f".envファイルが見つかりません: {path}")
    values: Dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f".envの{line_number}行目が KEY=VALUE 形式ではありません。")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values


def read_text_exact(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            value = handle.read()
    except UnicodeDecodeError as exc:
        raise ValueError(f"UTF-8で読み取れません: {path}") from exc
    if not value.strip():
        raise ValueError(f"知識ファイルが空です: {path}")
    return value


def build_system_prompt(knowledge: str, template: str = "") -> str:
    """Insert the complete knowledge into a user-supplied prompt template."""
    if not template:
        return ""
    if KNOWLEDGE_PLACEHOLDER not in template:
        raise ValueError(
            f"システムプロンプトファイルに {KNOWLEDGE_PLACEHOLDER} がありません。"
        )
    return template.replace(KNOWLEDGE_PLACEHOLDER, knowledge)


def build_user_prompt(example: Example) -> str:
    lines = ["次の問題に回答してください。", "", f"Question: {example.question}", "", "Choices:"]
    lines.extend(
        f"({letter}) {choice}" for letter, choice in zip(CHOICE_LETTERS, example.choices)
    )
    lines.extend(
        [
            "",
            "回答は次の形式のJSONオブジェクト1個だけにしてください: {\"answer\":\"A\"}",
            "answerの値はA、B、C、Dのいずれか1文字です。説明、推論、Markdownは出力しないでください。",
        ]
    )
    return "\n".join(lines)


def resolve_dataset_path(explicit: Optional[str], repo_root: Path, split: str) -> Path:
    candidate_text = explicit or os.getenv("GPQA_DATA_FILE")
    if candidate_text:
        candidate = Path(candidate_text).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        if not candidate.is_file():
            raise ValueError(f"データファイルが見つかりません: {candidate}")
        return candidate.resolve()

    csv_candidate = repo_root / "dataset" / f"gpqa_{split}.csv"
    if csv_candidate.is_file():
        return csv_candidate
    archive_candidate = repo_root / "dataset.zip"
    if archive_candidate.is_file():
        return archive_candidate
    raise ValueError(
        "GPQAデータが見つかりません。--data-file または GPQA_DATA_FILE を指定してください。"
    )


def open_dataset_text(
    data_path: Path, split: str, password: str
) -> Tuple[TextIO, Optional[zipfile.ZipFile], str]:
    if data_path.suffix.lower() != ".zip":
        return data_path.open("r", encoding="utf-8-sig", newline=""), None, str(data_path)

    archive = zipfile.ZipFile(data_path)
    member = f"dataset/gpqa_{split}.csv"
    try:
        binary = archive.open(member, pwd=password.encode("utf-8"))
    except (KeyError, RuntimeError) as exc:
        archive.close()
        raise ValueError(
            f"ZIP内の {member} を開けません。splitまたはGPQA_DATASET_PASSWORDを確認してください。"
        ) from exc
    text = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
    return text, archive, f"{data_path}!/{member}"


def load_examples(data_path: Path, split: str, password: str, seed: int) -> Tuple[List[Example], str]:
    handle, archive, source = open_dataset_text(data_path, split, password)
    try:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            raise ValueError(f"GPQA CSVの必須列がありません: {', '.join(missing)}")

        rng = random.Random(seed)
        examples: List[Example] = []
        for row_index, row in enumerate(reader):
            labelled_choices = [
                (str(row["Incorrect Answer 1"]), False),
                (str(row["Incorrect Answer 2"]), False),
                (str(row["Incorrect Answer 3"]), False),
                (str(row["Correct Answer"]), True),
            ]
            rng.shuffle(labelled_choices)
            choices = tuple(item[0] for item in labelled_choices)
            correct_index = next(index for index, item in enumerate(labelled_choices) if item[1])
            question_id = str(row.get("Record ID") or row_index)
            examples.append(
                Example(
                    question_id=question_id,
                    question=str(row["Question"]),
                    choices=choices,  # type: ignore[arg-type]
                    correct_index=correct_index,
                )
            )
    finally:
        handle.close()
        if archive is not None:
            archive.close()
    if not examples:
        raise ValueError("GPQAデータに問題がありません。")
    return examples, source


def chat_completions_endpoint(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


def ollama_chat_endpoint(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/api/chat"):
        return normalized
    return normalized + "/api/chat"


def extract_message_content(response: Dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ApiError("APIレスポンスに choices[0].message.content がありません。") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        if parts:
            return "".join(parts)
    raise ApiError("APIレスポンスのmessage.contentが文字列ではありません。")


def call_openai_compatible(
    base_url: str,
    api_key: Optional[str],
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: float,
    max_tokens: int,
    temperature: Optional[float],
    retries: int,
) -> Tuple[str, Dict[str, Any], Optional[str]]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }
    if not base_url.rstrip("/").startswith("https://ollama.com"):
        payload["max_tokens"] = max_tokens
    if temperature is not None and not base_url.rstrip("/").startswith("https://ollama.com"):
        payload["temperature"] = temperature
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    endpoint = chat_completions_endpoint(base_url)

    for attempt in range(retries + 1):
        client_request_id = str(uuid.uuid4())
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "gpqa-knowledge-eval/1.0",
            "X-Client-Request-Id": client_request_id,
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
                request_id = response.headers.get("x-request-id")
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ApiError("APIレスポンスの最上位がJSONオブジェクトではありません。")
            return extract_message_content(parsed), parsed.get("usage") or {}, request_id
        except urllib.error.HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
            retryable = exc.code in (408, 409, 429) or exc.code >= 500
            error = ApiError(f"HTTP {exc.code}: {detail}", status=exc.code)
            if not retryable or attempt >= retries:
                raise error
        except (urllib.error.URLError, TimeoutError) as exc:
            error = ApiError(f"API接続エラー: {exc}")
            if attempt >= retries:
                raise error
        except json.JSONDecodeError as exc:
            raise ApiError("APIレスポンスが有効なJSONではありません。") from exc
        time.sleep(min(2 ** attempt, 30))
    raise AssertionError("unreachable")


def call_ollama(
    base_url: str,
    api_key: Optional[str],
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: float,
    temperature: Optional[float],
    seed: int,
    retries: int,
) -> Tuple[str, Dict[str, Any], Optional[str]]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }
    # Ollama Cloud's direct API is kept to the documented minimal payload.
    # Local Ollama accepts generation controls through the options object.
    if not base_url.rstrip("/").startswith("https://ollama.com"):
        options: Dict[str, Any] = {"seed": seed}
        if temperature is not None:
            options["temperature"] = temperature
        payload["options"] = options
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    endpoint = ollama_chat_endpoint(base_url)

    for attempt in range(retries + 1):
        client_request_id = str(uuid.uuid4())
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "gpqa-knowledge-eval/1.0",
            "X-Client-Request-Id": client_request_id,
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
                request_id = response.headers.get("x-request-id")
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ApiError("Ollamaレスポンスの最上位がJSONオブジェクトではありません。")
            message = parsed.get("message")
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                raise ApiError("Ollamaレスポンスに message.content がありません。")
            usage = {
                key: parsed[key]
                for key in (
                    "prompt_eval_count",
                    "eval_count",
                    "total_duration",
                    "load_duration",
                    "prompt_eval_duration",
                    "eval_duration",
                )
                if isinstance(parsed.get(key), int)
            }
            return message["content"], usage, request_id
        except urllib.error.HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
            retryable = exc.code in (408, 409, 429) or exc.code >= 500
            error = ApiError(f"HTTP {exc.code}: {detail}", status=exc.code)
            if not retryable or attempt >= retries:
                raise error
        except (urllib.error.URLError, TimeoutError) as exc:
            error = ApiError(f"Ollama API接続エラー: {exc}")
            if attempt >= retries:
                raise error
        except json.JSONDecodeError as exc:
            raise ApiError("Ollama APIレスポンスが有効なJSONではありません。") from exc
        time.sleep(min(2 ** attempt, 30))
    raise AssertionError("unreachable")


def parse_json_answer(response_text: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    text = response_text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None, None
    if not isinstance(parsed, dict) or set(parsed) != {"answer"}:
        return None, parsed if isinstance(parsed, dict) else None
    answer = parsed.get("answer")
    if not isinstance(answer, str) or answer.upper() not in CHOICE_LETTERS:
        return None, parsed
    return answer.upper(), parsed


def parse_answer(response_text: str) -> Optional[str]:
    patterns = (
        r"(?im)^\s*FINAL[_ ]ANSWER\s*[:：=-]\s*\(?\s*([ABCD])\s*\)?\s*[.!。]?\s*$",
        r"(?i)(?:the\s+)?correct\s+answer\s+is\s*\(?\s*([ABCD])\s*\)?",
        r"(?im)^\s*(?:answer|回答)\s*[:：=-]\s*\(?\s*([ABCD])\s*\)?\s*[.!。]?\s*$",
    )
    for pattern in patterns:
        matches = re.findall(pattern, response_text)
        if matches:
            return matches[-1].upper()
    tail_match = re.search(r"(?i)\(?\s*([ABCD])\s*\)?\s*[.!。]?\s*$", response_text.strip())
    return tail_match.group(1).upper() if tail_match else None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def jsonl_record(value: Dict[str, Any]) -> str:
    """Serialize one LF-delimited JSON record without Unicode line separators."""
    serialized = json.dumps(value, ensure_ascii=False)
    for separator in ("\u0085", "\u2028", "\u2029"):
        serialized = serialized.replace(separator, f"\\u{ord(separator):04x}")
    return serialized + "\n"


def default_output_dir(split: str) -> Path:
    """Return a new timestamped run directory below the current directory."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path.cwd() / "run" / f"{split}-{stamp}"


def run(args: argparse.Namespace) -> int:
    script_path = Path(__file__).resolve()
    workspace_root = script_path.parents[1]
    gpqa_root = workspace_root / "GPQA"
    if not gpqa_root.is_dir():
        raise ValueError(f"GPQAリポジトリが見つかりません: {gpqa_root}")
    knowledge_path = Path(args.knowledge_file).expanduser().resolve()
    if not knowledge_path.is_file():
        raise ValueError(f"知識ファイルが見つかりません: {knowledge_path}")
    knowledge = read_text_exact(knowledge_path)
    system_prompt_path: Optional[Path] = None
    system_prompt_template = ""
    if args.system_prompt_file:
        system_prompt_path = Path(args.system_prompt_file).expanduser().resolve()
        if not system_prompt_path.is_file():
            raise ValueError(
                f"システムプロンプトファイルが見つかりません: {system_prompt_path}"
            )
        system_prompt_template = read_text_exact(system_prompt_path)
    system_prompt = build_system_prompt(knowledge, system_prompt_template)

    password = os.getenv("GPQA_DATASET_PASSWORD", OFFICIAL_ARCHIVE_PASSWORD)
    data_path = resolve_dataset_path(args.data_file, gpqa_root, args.split)
    examples, dataset_source = load_examples(data_path, args.split, password, args.seed)
    if args.max_examples is not None:
        if args.max_examples < 1:
            raise ValueError("--max-examples は1以上にしてください。")
        examples = examples[: args.max_examples]
    if args.max_tokens < 1:
        raise ValueError("--max-tokens は1以上にしてください。")
    if args.timeout <= 0:
        raise ValueError("--timeout は0より大きくしてください。")
    if args.retries < 0:
        raise ValueError("--retries は0以上にしてください。")
    if args.request_delay < 0:
        raise ValueError("--request-delay は0以上にしてください。")

    env_file = Path(args.env_file).expanduser()
    if not env_file.is_absolute():
        env_file = workspace_root / env_file
    env_values = load_env_file(env_file)
    api_style = args.api_style
    if api_style == "ollama":
        model = args.model or env_values.get("OLLAMA_MODEL") or DEFAULT_MODEL
        base_url = args.base_url or env_values.get("OLLAMA_BASE_URL") or DEFAULT_BASE_URL
    else:
        model = args.model or env_values.get("OPENAI_MODEL") or os.getenv("OPENAI_MODEL")
        base_url = (
            args.base_url
            or env_values.get("OPENAI_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or OPENAI_DEFAULT_BASE_URL
        )
    api_key = env_values.get(args.api_key_env) or os.getenv(args.api_key_env)
    metadata: Dict[str, Any] = {
        "mode": "dry-run" if args.dry_run else "evaluation",
        "dataset_source": dataset_source,
        "split": args.split,
        "examples": len(examples),
        "seed": args.seed,
        "knowledge_file": str(knowledge_path),
        "knowledge_chars": len(knowledge),
        "knowledge_sha256": sha256_text(knowledge),
        "system_prompt_file": str(system_prompt_path) if system_prompt_path else None,
        "system_prompt_template_chars": len(system_prompt_template),
        "system_prompt_template_sha256": sha256_text(system_prompt_template),
        "system_prompt_chars": len(system_prompt),
        "system_prompt_sha256": sha256_text(system_prompt),
        "knowledge_is_complete_substring": knowledge in system_prompt,
        "api_style": api_style,
        "model": model,
        "base_url": base_url,
        "temperature_requested": args.temperature,
        "generation_options_sent": not base_url.rstrip("/").startswith("https://ollama.com"),
    }
    if args.dry_run:
        metadata["first_question_id"] = examples[0].question_id
        metadata["first_correct_letter"] = CHOICE_LETTERS[examples[0].correct_index]
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0

    if not model:
        raise ValueError("モデル名を --model または.envで指定してください。")
    if not api_key and base_url.startswith("https://"):
        raise ValueError(f"リモートAPIには.envの {args.api_key_env} が必要です。")

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = default_output_dir(args.split)
    output_dir.mkdir(parents=True, exist_ok=args.overwrite_output)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"

    correct = 0
    parsed_count = 0
    errors = 0
    token_totals: Dict[str, int] = {}
    started_at = utc_now()
    with results_path.open("w", encoding="utf-8", newline="\n") as result_file:
        for index, example in enumerate(examples):
            user_prompt = build_user_prompt(example)
            record: Dict[str, Any] = {
                "index": index,
                "question_id": example.question_id,
                "question": example.question,
                "choices": dict(zip(CHOICE_LETTERS, example.choices)),
                "correct_letter": CHOICE_LETTERS[example.correct_index],
                "correct_answer": example.choices[example.correct_index],
            }
            try:
                if api_style == "ollama":
                    response_text, usage, request_id = call_ollama(
                        base_url=base_url,
                        api_key=api_key,
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        timeout=args.timeout,
                        temperature=args.temperature,
                        seed=args.seed,
                        retries=args.retries,
                    )
                    predicted, response_json = parse_json_answer(response_text)
                else:
                    response_text, usage, request_id = call_openai_compatible(
                        base_url=base_url,
                        api_key=api_key,
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        timeout=args.timeout,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        retries=args.retries,
                    )
                    predicted, response_json = parse_json_answer(response_text)
                is_correct = predicted == record["correct_letter"]
                if predicted is not None:
                    parsed_count += 1
                if is_correct:
                    correct += 1
                record.update(
                    {
                        "predicted_letter": predicted,
                        "is_correct": is_correct,
                        "response": response_text,
                        "response_json": response_json,
                        "response_format_valid": predicted is not None,
                        "usage": usage,
                        "request_id": request_id,
                        "error": None,
                    }
                )
                for key, value in usage.items():
                    if isinstance(value, int):
                        token_totals[key] = token_totals.get(key, 0) + value
            except ApiError as exc:
                errors += 1
                record.update(
                    {
                        "predicted_letter": None,
                        "is_correct": False,
                        "response": None,
                        "response_json": None,
                        "response_format_valid": False,
                        "usage": {},
                        "request_id": None,
                        "error": str(exc),
                    }
                )
                if args.fail_fast:
                    result_file.write(jsonl_record(record))
                    result_file.flush()
                    raise
            result_file.write(jsonl_record(record))
            result_file.flush()
            print(
                f"[{index + 1}/{len(examples)}] id={example.question_id} "
                f"answer={record['predicted_letter'] or '-'} correct={record['is_correct']}",
                file=sys.stderr,
            )
            if args.request_delay > 0 and index + 1 < len(examples):
                time.sleep(args.request_delay)

    unparsed_count = len(examples) - parsed_count - errors
    summary: Dict[str, Any] = dict(metadata)
    summary.update(
        {
            "started_at": started_at,
            "finished_at": utc_now(),
            "output_dir": str(output_dir),
            "total": len(examples),
            "correct": correct,
            "accuracy": correct / len(examples),
            "parsed": parsed_count,
            "unparsed": unparsed_count,
            "api_errors": errors,
            "parsed_accuracy": correct / parsed_count if parsed_count else None,
            "usage_totals": token_totals,
        }
    )
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if errors == 0 and unparsed_count == 0 else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "指定時は知識テキスト全文をsystem promptテンプレートへ埋め込み、"
            "GPQAをOllama Cloud APIで評価します。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("knowledge_file", help="UTF-8の知識テキストファイル")
    parser.add_argument(
        "--system-prompt-file",
        help=(
            f"UTF-8のsystem promptテンプレート。{KNOWLEDGE_PLACEHOLDER} を知識全文へ置換"
            "（未指定時はsystem promptが空）"
        ),
    )
    parser.add_argument(
        "--data-file",
        help="GPQA CSVまたは公式dataset.zip（未指定時はGPQA_DATA_FILE、dataset/、dataset.zipの順）",
    )
    parser.add_argument(
        "--split", choices=("main", "diamond", "experts", "extended"), default="main"
    )
    parser.add_argument("--api-style", choices=("ollama", "openai"), default="ollama")
    parser.add_argument("--model", help=f"モデル名（Ollamaの既定値は{DEFAULT_MODEL}）")
    parser.add_argument("--base-url", help=f"APIのbase URL（Ollamaの既定値は{DEFAULT_BASE_URL}）")
    parser.add_argument("--env-file", default=".env", help="APIキー等を読むdotenvファイル")
    parser.add_argument(
        "--api-key-env",
        default="API_KEY",
        help=".envまたは環境変数からAPIキーを読むキー名（値自体はCLIへ渡さない）",
    )
    parser.add_argument("--max-examples", type=int, help="先頭から評価する問題数")
    parser.add_argument("--seed", type=int, default=0, help="選択肢シャッフルのseed")
    parser.add_argument("--max-tokens", type=int, default=1024, help="回答の最大生成token数")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="ローカルOllamaまたはOpenAI互換APIへ送る生成temperature（Cloud直結では省略）",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="1リクエストのtimeout秒")
    parser.add_argument("--retries", type=int, default=3, help="一時的エラーの再試行回数")
    parser.add_argument("--request-delay", type=float, default=0.0, help="問題間の待機秒")
    parser.add_argument(
        "--output-dir",
        help="結果ディレクトリ（未指定時はカレント直下のrun/、既存ディレクトリは拒否）",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="既存ディレクトリ内のresults.jsonlとsummary.jsonを上書き",
    )
    parser.add_argument("--fail-fast", action="store_true", help="最初のAPIエラーで停止")
    parser.add_argument("--dry-run", action="store_true", help="APIを呼ばず、入力・prompt構築だけ検証")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ApiError, OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
