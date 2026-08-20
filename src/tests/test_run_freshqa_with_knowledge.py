import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_freshqa_with_knowledge.py"
SPEC = importlib.util.spec_from_file_location("freshqa_knowledge", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_official_style_csv(path: Path, questions=1) -> None:
    headers = [
        "id",
        "question",
        "answer_0",
        "answer_1",
        "fact_type",
        "num_hops",
        "false_premise",
        "split",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["FreshQA benchmark snapshot"])
        writer.writerow(["Downloaded for testing"])
        writer.writerow(headers)
        for index in range(questions):
            writer.writerow(
                [
                    f"q-{index}",
                    f"Question {index}?",
                    f"Answer {index}",
                    f"Alternative {index}",
                    "fast-changing",
                    "one-hop",
                    "FALSE",
                    "TEST",
                ]
            )


class PromptAndDatasetTests(unittest.TestCase):
    def test_default_ollama_judge_model_is_qwen_38_27b(self):
        self.assertEqual(MODULE.DEFAULT_OLLAMA_JUDGE_MODEL, "qwen3.8:27b")
        self.assertEqual(
            MODULE.DEFAULT_OLLAMA_JUDGE_BASE_URL, "http://localhost:11434"
        )

    def test_system_prompt_contains_complete_knowledge_and_instruction(self):
        knowledge = "first line\r\nsecond line\n<xml-like>value</xml-like>"
        prompt = MODULE.build_system_prompt(knowledge)
        self.assertIn(knowledge, prompt)
        self.assertIn("質問回答のために利用可能な知識", prompt)
        self.assertIn("利用可能な知識を抽出して利用", prompt)

    def test_custom_system_prompt_replaces_knowledge_placeholder(self):
        knowledge = "complete\r\nknowledge"
        prompt = MODULE.build_system_prompt(
            knowledge, "Custom instructions\n{{knowledge}}\nEnd"
        )
        self.assertEqual(prompt, f"Custom instructions\n{knowledge}\nEnd")
        self.assertNotIn("{{knowledge}}", prompt)

    def test_custom_system_prompt_requires_knowledge_placeholder(self):
        with self.assertRaisesRegex(ValueError, "\\{\\{knowledge\\}\\}"):
            MODULE.build_system_prompt("knowledge", "Custom instructions only")

    def test_official_style_csv_header_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "freshqa.csv"
            write_official_style_csv(path)
            examples, schema = MODULE.load_examples(path)
        self.assertEqual(schema["header_row"], 3)
        self.assertEqual(schema["answer_columns"], ["answer_0", "answer_1"])
        self.assertEqual(examples[0].question_id, "q-0")
        self.assertEqual(
            examples[0].correct_answers, ("Answer 0", "Alternative 0")
        )
        self.assertEqual(examples[0].metadata["split"], "TEST")

    def test_current_gviz_warning_header_and_empty_columns_are_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "freshqa.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "Warning: do not rate these examples. id",
                        "split",
                        "question",
                        "answer_0",
                        "",
                        "",
                    ]
                )
                writer.writerow(["7", "TEST", "Current question?", "Current answer", "", ""])
            examples, schema = MODULE.load_examples(path)
        self.assertEqual(schema["header_row"], 1)
        self.assertEqual(schema["columns"][0], "id")
        self.assertEqual(examples[0].question_id, "7")
        self.assertEqual(examples[0].correct_answers, ("Current answer",))

    def test_split_filter_is_case_insensitive(self):
        example = MODULE.FreshQAExample(
            question_id="1",
            question="Q?",
            correct_answers=("A",),
            metadata={"split": "TEST"},
        )
        self.assertEqual(MODULE.apply_split_filter([example], "test"), [example])

    def test_default_dataset_is_resolved_under_src(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace_root = Path(directory)
            data_path = workspace_root / "src" / "data" / "freshqa.csv"
            data_path.parent.mkdir(parents=True)
            data_path.write_text("placeholder", encoding="utf-8")
            resolved = MODULE.resolve_dataset_path(None, workspace_root)
        self.assertEqual(resolved, data_path)

    def test_judge_response_parser_accepts_json_and_fence(self):
        self.assertEqual(
            MODULE.parse_judge_response(
                '{"rating":"TRUE","explanation":"matches"}'
            )[:2],
            (True, "matches"),
        )
        self.assertFalse(
            MODULE.parse_judge_response(
                '```json\n{"rating":"incorrect","explanation":"no"}\n```'
            )[0]
        )
        self.assertIsNone(MODULE.parse_judge_response("evaluation: correct")[0])


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content
        self.headers = {"x-request-id": "req-test"}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(
            {
                "choices": [{"message": {"content": self.content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            }
        ).encode("utf-8")


class ApiTests(unittest.TestCase):
    def test_openai_compatible_request_preserves_system_knowledge(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return _FakeResponse("answer")

        messages = [
            {"role": "system", "content": "SYSTEM WITH COMPLETE KNOWLEDGE"},
            {"role": "user", "content": "QUESTION"},
        ]
        with mock.patch.object(
            MODULE.urllib.request, "urlopen", side_effect=fake_urlopen
        ):
            text, usage, request_id = MODULE.call_openai_compatible(
                base_url="http://localhost:8000/v1",
                api_key="test-key",
                model="test-model",
                messages=messages,
                timeout=2,
                max_tokens=64,
                temperature=0.0,
                retries=0,
                purpose="test",
            )
        received = json.loads(captured["request"].data.decode("utf-8"))
        self.assertEqual(text, "answer")
        self.assertEqual(usage["prompt_tokens"], 10)
        self.assertEqual(request_id, "req-test")
        self.assertEqual(received["messages"], messages)
        self.assertEqual(received["max_tokens"], 64)
        self.assertEqual(
            captured["request"].get_header("Authorization"), "Bearer test-key"
        )

    def test_ollama_cloud_request_uses_api_chat_and_minimal_payload(self):
        captured = {}

        class FakeOllamaResponse(_FakeResponse):
            def read(self):
                return json.dumps(
                    {
                        "message": {"role": "assistant", "content": "answer"},
                        "prompt_eval_count": 20,
                        "eval_count": 5,
                    }
                ).encode("utf-8")

        def fake_urlopen(request, timeout):
            captured["request"] = request
            return FakeOllamaResponse("unused")

        messages = [
            {"role": "system", "content": "SYSTEM WITH COMPLETE KNOWLEDGE"},
            {"role": "user", "content": "QUESTION"},
        ]
        with mock.patch.object(
            MODULE.urllib.request, "urlopen", side_effect=fake_urlopen
        ):
            text, usage, _ = MODULE.call_ollama(
                base_url="https://ollama.com",
                api_key="test-key",
                model="gpt-oss:120b",
                messages=messages,
                timeout=2,
                temperature=0.0,
                retries=0,
                purpose="test",
            )
        received = json.loads(captured["request"].data.decode("utf-8"))
        self.assertEqual(
            captured["request"].full_url, "https://ollama.com/api/chat"
        )
        self.assertEqual(received["model"], "gpt-oss:120b")
        self.assertEqual(received["messages"], messages)
        self.assertNotIn("max_tokens", received)
        self.assertNotIn("temperature", received)
        self.assertNotIn("options", received)
        self.assertEqual(text, "answer")
        self.assertEqual(usage, {"prompt_eval_count": 20, "eval_count": 5})


class EvaluationTests(unittest.TestCase):
    def test_run_answers_judges_and_writes_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            knowledge_path = root / "knowledge.txt"
            knowledge_path.write_text("complete knowledge", encoding="utf-8")
            data_path = root / "freshqa.csv"
            write_official_style_csv(data_path)
            env_path = root / ".env"
            env_path.write_text(
                "OPENAI_API_KEY=test-key\nOPENAI_MODEL=test-model\n",
                encoding="utf-8",
            )
            output_dir = root / "output"
            prompt_path = root / "system-prompt.txt"
            prompt_path.write_text(
                "Custom FreshQA instructions\n{{knowledge}}", encoding="utf-8"
            )
            args = MODULE.build_parser().parse_args(
                [
                    str(knowledge_path),
                    "--data-file",
                    str(data_path),
                    "--env-file",
                    str(env_path),
                    "--system-prompt-file",
                    str(prompt_path),
                    "--api-style",
                    "openai",
                    "--judge-model",
                    "test-model",
                    "--output-dir",
                    str(output_dir),
                ]
            )
            fake_calls = [
                ("Answer 0", {"prompt_tokens": 7}, "answer-req"),
                (
                    '{"rating":"TRUE","explanation":"reference matched"}',
                    {"prompt_tokens": 11},
                    "judge-req",
                ),
            ]
            with mock.patch.object(MODULE, "call_model_api", side_effect=fake_calls) as call:
                self.assertEqual(MODULE.run(args), 0)

            self.assertEqual(call.call_count, 2)
            answer_messages = call.call_args_list[0].kwargs["messages"]
            self.assertIn("complete knowledge", answer_messages[0]["content"])
            judge_messages = call.call_args_list[1].kwargs["messages"]
            self.assertIn("Answer 0", judge_messages[1]["content"])

            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            result = json.loads(
                (output_dir / "results.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["total"], 1)
            self.assertEqual(summary["judged"], 1)
            self.assertEqual(summary["correct"], 1)
            self.assertEqual(summary["accuracy"], 1.0)
            self.assertEqual(summary["answer_usage_totals"], {"prompt_tokens": 7})
            self.assertEqual(summary["judge_usage_totals"], {"prompt_tokens": 11})
            self.assertTrue(summary["knowledge_is_complete_substring"])
            self.assertEqual(summary["system_prompt_source"], "file")
            self.assertEqual(summary["system_prompt_file"], str(prompt_path.resolve()))
            self.assertTrue(result["is_correct"])
            self.assertEqual(result["question_id"], "q-0")

    def test_dry_run_does_not_require_model_or_api_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            knowledge_path = root / "knowledge.txt"
            knowledge_path.write_text("knowledge", encoding="utf-8")
            data_path = root / "freshqa.csv"
            write_official_style_csv(data_path, questions=2)
            args = MODULE.build_parser().parse_args(
                [
                    str(knowledge_path),
                    "--data-file",
                    str(data_path),
                    "--env-file",
                    str(root / "missing.env"),
                    "--dry-run",
                    "--max-examples",
                    "1",
                ]
            )
            with mock.patch.object(MODULE, "call_model_api") as call:
                self.assertEqual(MODULE.run(args), 0)
                call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
