import json
import os
import sys
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from pydantic import ValidationError

from crynux_worker.model import (
    DownloadTaskInput,
    InferenceTaskInput,
    ModelFileEntry,
    TaskInput,
    TaskResult,
    TaskType,
)
from crynux_worker.model_cache import ModelCache
from crynux_worker.task.download import _download_one_task
from crynux_worker.task.inference import _inference_one_task
from crynux_worker.task.runner import MOCK_REVISION, HFTaskRunner, MockTaskRunner

SHA = "a" * 40
OTHER_SHA = "b" * 40
CACHE_HASH = "c" * 64
LLM_TASK_ARGS = json.dumps(
    {"model": "org/model", "messages": [{"role": "user", "content": "hi"}]}
)


def entry(repository="org/model", revision="", model_format="", format_kwargs=None):
    return ModelFileEntry(
        repository=repository,
        revision=revision,
        format=model_format,
        format_kwargs=format_kwargs or {},
    )


class InputModelTests(unittest.TestCase):
    def test_download_input_accepts_model_files(self):
        task_input = TaskInput.model_validate(
            {
                "task": {
                    "task_name": "download",
                    "task_type": 1,
                    "task_id": "task",
                    "model_files": [
                        {
                            "source_type": "huggingface",
                            "repository": "org/model",
                            "format": "",
                            "format_kwargs": None,
                        }
                    ],
                }
            }
        ).task

        self.assertIsInstance(task_input, DownloadTaskInput)
        self.assertIsNone(task_input.model)
        self.assertEqual(task_input.model_files, [entry()])
        self.assertFalse(task_input.model_files[0].is_exact)

    def test_download_input_accepts_legacy_model(self):
        task_input = DownloadTaskInput.model_validate(
            {
                "task_name": "download",
                "task_type": 0,
                "task_id": "task",
                "model": {"id": "org/lora", "type": "lora"},
            }
        )

        self.assertEqual(task_input.model_files, [])

    def test_download_input_requires_exactly_one_target(self):
        base = {"task_name": "download", "task_type": 1, "task_id": "task"}
        with self.assertRaises(ValidationError):
            DownloadTaskInput.model_validate(base)
        with self.assertRaises(ValidationError):
            DownloadTaskInput.model_validate(
                {
                    **base,
                    "model": {"id": "org/model", "type": "base"},
                    "model_files": [{"repository": "org/model"}],
                }
            )

    def test_inference_input_accepts_execution_model_files(self):
        task_input = InferenceTaskInput.model_validate(
            {
                "task_name": "inference",
                "task_type": 1,
                "task_id": "task",
                "models": [{"id": "org/model", "type": "base"}],
                "task_args": "{}",
                "output_dir": "output",
                "execution_model_files": [
                    {"repository": "org/model", "revision": SHA}
                ],
                "requested_model_cache_hash": CACHE_HASH,
            }
        )

        self.assertEqual(task_input.execution_model_files, [entry(revision=SHA)])
        self.assertEqual(task_input.requested_model_cache_hash, CACHE_HASH)

    def test_inference_input_defaults_without_model_files(self):
        task_input = InferenceTaskInput.model_validate(
            {
                "task_name": "inference",
                "task_type": 1,
                "task_id": "task",
                "models": [],
                "task_args": "{}",
                "output_dir": "output",
            }
        )

        self.assertEqual(task_input.execution_model_files, [])
        self.assertIsNone(task_input.requested_model_cache_hash)

    def test_download_result_wire_format(self):
        result = TaskResult.model_validate(
            {
                "task_name": "download",
                "task_id_commitment": "task",
                "result": {
                    "status": "success",
                    "model_files": [
                        entry(revision=SHA, model_format="gguf").model_dump()
                    ],
                },
            }
        )
        payload = json.loads(result.model_dump_json(exclude_none=True))

        self.assertEqual(
            payload["result"]["model_files"],
            [
                {
                    "source_type": "huggingface",
                    "repository": "org/model",
                    "revision": SHA,
                    "format": "gguf",
                    "format_kwargs": {},
                }
            ],
        )
        self.assertNotIn("model_cache_hash", payload["result"])


class ModelCacheTests(unittest.TestCase):
    def test_delete_clears_matching_key(self):
        cache = ModelCache()
        cache.load("key", lambda: "model")

        cache.delete("key")

        self.assertIsNone(cache.key)
        self.assertIsNone(cache.value)

    def test_delete_keeps_other_key(self):
        cache = ModelCache()
        cache.load("key", lambda: "model")

        cache.delete("other")

        self.assertEqual(cache.key, "key")
        self.assertEqual(cache.value, "model")


class InferenceCoordinatorTests(unittest.TestCase):
    def _run(self, task_type, requested_model_cache_hash):
        task_runner = MagicMock()
        task_runner.inference.return_value = {}
        model_files = [entry(revision=SHA)]
        task_input = SimpleNamespace(
            task_type=task_type,
            task_args="{}",
            execution_model_files=model_files,
            requested_model_cache_hash=requested_model_cache_hash,
        )
        with patch("crynux_worker.task.inference.prepare_gpu_for_task"):
            result = _inference_one_task(
                task_runner=task_runner,
                task_input=task_input,
                model_cache=MagicMock(),
                sd_config=MagicMock(),
                gpt_config=MagicMock(),
                output_dir="output",
            )
        return task_runner, model_files, result

    def test_echoes_requested_model_cache_hash(self):
        for task_type in (TaskType.LLM, TaskType.SD):
            with self.subTest(task_type=task_type):
                _, _, result = self._run(task_type, CACHE_HASH)

                self.assertEqual(result.status, "success")
                self.assertEqual(result.model_cache_hash, CACHE_HASH)

    def test_passes_execution_model_files_to_runner(self):
        task_runner, model_files, _ = self._run(TaskType.LLM, CACHE_HASH)

        self.assertEqual(
            task_runner.inference.call_args.kwargs["execution_model_files"],
            model_files,
        )

    def test_omits_hash_without_request(self):
        _, _, result = self._run(TaskType.LLM, None)

        self.assertIsNone(result.model_cache_hash)

    def test_finetune_does_not_report_cache_hash(self):
        _, _, result = self._run(TaskType.SD_FT_LORA, CACHE_HASH)

        self.assertIsNone(result.model_cache_hash)


class RunnerEnforcementTests(unittest.TestCase):
    def _run_llm(self, task_args, execution_model_files):
        with (
            tempfile.TemporaryDirectory() as output_dir,
            patch.dict(os.environ, {}, clear=True),
            patch("gpt_task.inference.run_task", return_value={}) as run_task,
        ):
            HFTaskRunner().inference(
                task_type=TaskType.LLM,
                task_args=task_args,
                execution_model_files=execution_model_files,
                model_cache=MagicMock(),
                sd_config=MagicMock(),
                gpt_config=MagicMock(),
                output_dir=output_dir,
            )
        return run_task

    def test_llm_runs_with_allocated_revision(self):
        run_task = self._run_llm(LLM_TASK_ARGS, [entry(revision=SHA)])

        self.assertEqual(run_task.call_args.args[0].revision, SHA)

    def test_llm_rejects_conflicting_task_revision(self):
        task_args = json.dumps({**json.loads(LLM_TASK_ARGS), "revision": OTHER_SHA})

        with self.assertRaises(ValueError) as ctx:
            self._run_llm(task_args, [entry(revision=SHA)])

        self.assertIn("Task execution error", str(ctx.exception))
        self.assertNotIn("Task args invalid", str(ctx.exception))

    def test_llm_rejects_missing_allocation(self):
        with self.assertRaises(ValueError) as ctx:
            self._run_llm(LLM_TASK_ARGS, [])

        self.assertIn("Task execution error", str(ctx.exception))

class DownloadFileTests(unittest.TestCase):
    def _sd_modules(self):
        download_module = ModuleType("sd_task.download_model")
        download_module.check_and_download_hf_pipeline = MagicMock()
        download_module.requests_proxy_session = MagicMock()
        component_module = ModuleType("sd_task.task_runner.image_task.download_model")
        component_module.download_component_file = MagicMock()
        modules = {
            "sd_task.download_model": download_module,
            "sd_task.task_runner.image_task.download_model": component_module,
        }
        return modules, download_module, component_module

    def test_llm_exact_revision_downloads_that_revision(self):
        with (
            patch("gpt_task.prefetch.download_model") as download_model,
            patch("huggingface_hub.model_info") as model_info,
        ):
            result = HFTaskRunner().download_model_file(
                TaskType.LLM,
                entry(revision=SHA),
                MagicMock(),
                MagicMock(),
            )

        model_info.assert_not_called()
        self.assertEqual(download_model.call_args.args[0], "org/model")
        self.assertEqual(download_model.call_args.args[3], SHA)
        self.assertEqual(result, entry(revision=SHA))

    def test_llm_empty_revision_resolves_commit_sha(self):
        with (
            patch("gpt_task.prefetch.download_model") as download_model,
            patch(
                "huggingface_hub.model_info", return_value=SimpleNamespace(sha=SHA)
            ) as model_info,
        ):
            result = HFTaskRunner().download_model_file(
                TaskType.LLM,
                entry(),
                MagicMock(),
                MagicMock(),
            )

        model_info.assert_called_once_with("org/model")
        self.assertEqual(download_model.call_args.args[3], SHA)
        self.assertEqual(result, entry(revision=SHA))

    def test_llm_rejects_unsupported_format(self):
        with patch("gpt_task.prefetch.download_model") as download_model:
            with self.assertRaises(ValueError):
                HFTaskRunner().download_model_file(
                    TaskType.LLM,
                    entry(revision=SHA, model_format="gguf"),
                    MagicMock(),
                    MagicMock(),
                )

        download_model.assert_not_called()

    def test_rejects_unsupported_source_type(self):
        with self.assertRaises(ValueError):
            HFTaskRunner().download_model_file(
                TaskType.LLM,
                ModelFileEntry(source_type="other", repository="org/model"),
                MagicMock(),
                MagicMock(),
            )

    def test_image_pipeline_downloads_resolved_revision(self):
        modules, download_module, component_module = self._sd_modules()
        with (
            patch.dict(sys.modules, modules),
            patch(
                "huggingface_hub.model_info", return_value=SimpleNamespace(sha=SHA)
            ),
            patch(
                "huggingface_hub.list_repo_files",
                return_value=["model_index.json", "vae/config.json"],
            ),
        ):
            result = HFTaskRunner().download_model_file(
                TaskType.SD,
                entry("org/base", model_format="diffusers"),
                MagicMock(),
                MagicMock(),
            )

        call = download_module.check_and_download_hf_pipeline.call_args
        self.assertEqual(call.args[0], "org/base")
        self.assertEqual(call.kwargs["revision"], SHA)
        self.assertEqual(result, entry("org/base", SHA, "diffusers"))
        component_module.download_component_file.assert_not_called()

    def test_gguf_component_downloads_exact_revision(self):
        modules, _, component_module = self._sd_modules()
        file_entry = entry(
            "org/gguf", SHA, "gguf", {"filename": "model.gguf"}
        )
        with patch.dict(sys.modules, modules):
            result = HFTaskRunner().download_model_file(
                TaskType.SD, file_entry, MagicMock(), MagicMock()
            )

        source = component_module.download_component_file.call_args.args[0]
        self.assertEqual(source.repo, "org/gguf")
        self.assertEqual(source.revision, SHA)
        self.assertEqual(source.format_kwargs, {"filename": "model.gguf"})
        self.assertEqual(result, file_entry)

    def test_diffusers_component_downloads_without_model_index(self):
        modules, download_module, component_module = self._sd_modules()
        file_entry = entry("org/vae", SHA, "diffusers", {})
        with (
            patch.dict(sys.modules, modules),
            patch(
                "huggingface_hub.list_repo_files",
                return_value=["config.json", "diffusion_pytorch_model.safetensors"],
            ),
        ):
            result = HFTaskRunner().download_model_file(
                TaskType.SD, file_entry, MagicMock(), MagicMock()
            )

        download_module.check_and_download_hf_pipeline.assert_not_called()
        source = component_module.download_component_file.call_args.args[0]
        self.assertEqual(source.repo, "org/vae")
        self.assertEqual(source.format, "diffusers")
        self.assertEqual(source.revision, SHA)
        self.assertEqual(result, file_entry)

    def test_image_rejects_empty_format(self):
        modules, download_module, component_module = self._sd_modules()
        with patch.dict(sys.modules, modules):
            with self.assertRaises(ValueError):
                HFTaskRunner().download_model_file(
                    TaskType.SD,
                    entry("org/base", SHA, ""),
                    MagicMock(),
                    MagicMock(),
                )

        download_module.check_and_download_hf_pipeline.assert_not_called()
        component_module.download_component_file.assert_not_called()

    def test_mock_runner_reports_concrete_revision(self):
        runner = MockTaskRunner()

        resolved = runner.download_model_file(
            TaskType.LLM, entry(), MagicMock(), MagicMock()
        )
        exact = runner.download_model_file(
            TaskType.LLM, entry(revision=SHA), MagicMock(), MagicMock()
        )

        self.assertEqual(resolved.revision, MOCK_REVISION)
        self.assertEqual(exact.revision, SHA)


class DownloadCoordinatorTests(unittest.TestCase):
    def test_model_files_result_lists_downloaded_entries(self):
        task_runner = MagicMock()
        task_runner.download_model_file.side_effect = (
            lambda task_type, model_file, sd_config, gpt_config: model_file.model_copy(
                update={"revision": SHA}
            )
        )
        task_input = DownloadTaskInput(
            task_name="download",
            task_type=TaskType.LLM,
            task_id="task",
            model_files=[entry("org/a"), entry("org/b")],
        )

        result = _download_one_task(
            task_runner, task_input, MagicMock(), MagicMock()
        )

        self.assertEqual(
            result.model_files,
            [entry("org/a", SHA), entry("org/b", SHA)],
        )
        task_runner.download_model.assert_not_called()

    def test_legacy_model_download_has_no_model_files(self):
        task_runner = MagicMock()
        task_input = DownloadTaskInput.model_validate(
            {
                "task_name": "download",
                "task_type": 0,
                "task_id": "task",
                "model": {"id": "org/lora", "type": "lora"},
            }
        )

        result = _download_one_task(
            task_runner, task_input, MagicMock(), MagicMock()
        )

        self.assertIsNone(result.model_files)
        task_runner.download_model.assert_called_once()
        task_runner.download_model_file.assert_not_called()


if __name__ == "__main__":
    unittest.main()
