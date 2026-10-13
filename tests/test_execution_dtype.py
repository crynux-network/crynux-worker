import json
import os
import sys
import tempfile
import unittest
from queue import Queue
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from crynux_worker.model import ModelFileEntry, SuccessResult, TaskResult, TaskType
from crynux_worker.task.inference import _inference_one_task
from crynux_worker.task.runner import HFTaskRunner
from crynux_worker.task.worker import TaskWorker

SHA = "a" * 40
LLM_TASK_ARGS = json.dumps(
    {"model": "org/model", "messages": [{"role": "user", "content": "hi"}]}
)
LLM_MODEL_FILES = [ModelFileEntry(repository="org/model", revision=SHA)]


def make_task_input(task_type=TaskType.LLM):
    return SimpleNamespace(
        task_type=task_type,
        task_args="{}",
        execution_model_files=[],
        requested_model_cache_hash=None,
    )


class DtypeList(list):
    def __init__(self, values, execution_dtype):
        super().__init__(values)
        self.execution_dtype = execution_dtype


class DtypeDict(dict):
    def __init__(self, values, execution_dtype):
        super().__init__(values)
        self.execution_dtype = execution_dtype


class ExecutionDtypeTests(unittest.TestCase):
    def test_success_result_accepts_legacy_payload(self):
        result = TaskResult.model_validate(
            {
                "task_name": "inference",
                "task_id_commitment": "task",
                "result": {"status": "success"},
            }
        )

        self.assertIsNone(result.result.execution_dtype)

    def test_success_result_accepts_dtype_payload(self):
        result = TaskResult.model_validate(
            {
                "task_name": "inference",
                "task_id_commitment": "task",
                "result": {
                    "status": "success",
                    "execution_dtype": "bfloat16",
                },
            }
        )

        self.assertEqual(result.result.execution_dtype, "bfloat16")

    def test_inference_coordinator_reads_runner_dtype(self):
        task_runner = MagicMock()
        task_runner.inference.return_value = DtypeDict({}, "float16")
        task_input = make_task_input()

        with patch("crynux_worker.task.inference.prepare_gpu_for_task"):
            result = _inference_one_task(
                task_runner=task_runner,
                task_input=task_input,
                model_cache=MagicMock(),
                sd_config=MagicMock(),
                gpt_config=MagicMock(),
                output_dir="output",
            )

        self.assertEqual(result.execution_dtype, "float16")

    def test_inference_coordinator_accepts_legacy_runner_result(self):
        task_runner = MagicMock()
        task_runner.inference.return_value = {}
        task_input = make_task_input()

        with patch("crynux_worker.task.inference.prepare_gpu_for_task"):
            result = _inference_one_task(
                task_runner=task_runner,
                task_input=task_input,
                model_cache=MagicMock(),
                sd_config=MagicMock(),
                gpt_config=MagicMock(),
                output_dir="output",
            )

        self.assertIsNone(result.execution_dtype)

    def test_sd_runner_preserves_dtype_result(self):
        image = MagicMock()
        runner_result = DtypeList([image], "float16")
        runner = HFTaskRunner()
        image_task_args = MagicMock()
        task_args_module = ModuleType("sd_task.task_args.image_task")
        task_args_module.ImageTaskArgs = image_task_args
        applied_args = MagicMock()
        runner_module = ModuleType("sd_task.task_runner.image_task")
        runner_module.run_image_task = MagicMock(return_value=runner_result)
        runner_module.apply_execution_model_files = MagicMock(
            return_value=applied_args
        )
        model_files = [ModelFileEntry(repository="org/base", revision=SHA)]

        with (
            tempfile.TemporaryDirectory() as output_dir,
            patch.dict(
                sys.modules,
                {
                    "sd_task.task_args.image_task": task_args_module,
                    "sd_task.task_runner.image_task": runner_module,
                },
            ),
        ):
            result = runner.inference(
                task_type=TaskType.SD,
                task_args="{}",
                execution_model_files=model_files,
                model_cache=MagicMock(),
                sd_config=MagicMock(),
                gpt_config=MagicMock(),
                output_dir=output_dir,
            )

            image.save.assert_called_once_with(os.path.join(output_dir, "0.png"))
            image_task_args.model_validate_json.assert_called_once_with("{}")
            runner_module.apply_execution_model_files.assert_called_once_with(
                image_task_args.model_validate_json.return_value, model_files
            )
            self.assertIs(
                runner_module.run_image_task.call_args.args[0], applied_args
            )

        self.assertIs(result, runner_result)

    def test_gpt_classic_runner_preserves_dtype_result(self):
        runner_response = {"choices": []}
        runner = HFTaskRunner()

        with (
            tempfile.TemporaryDirectory() as output_dir,
            patch.dict(os.environ, {}, clear=True),
            patch(
                "gpt_task.inference.run_task", return_value=runner_response
            ) as run_task,
            patch(
                "gpt_task.inference.get_execution_dtype",
                return_value="bfloat16",
            ),
        ):
            result = runner.inference(
                task_type=TaskType.LLM,
                task_args=LLM_TASK_ARGS,
                execution_model_files=LLM_MODEL_FILES,
                model_cache=MagicMock(),
                sd_config=MagicMock(),
                gpt_config=MagicMock(),
                output_dir=output_dir,
            )

            with open(
                os.path.join(output_dir, "0.json"), mode="r", encoding="utf-8"
            ) as f:
                self.assertEqual(json.load(f), {"choices": []})
            self.assertEqual(run_task.call_args.args[0].revision, SHA)

        self.assertEqual(result.execution_dtype, "bfloat16")

    def test_gpt_tp_runner_preserves_dtype_result(self):
        runner_response = {"choices": []}
        runner = HFTaskRunner()

        with (
            tempfile.TemporaryDirectory() as output_dir,
            patch.dict(os.environ, {"GPT_EXECUTOR": "tensor_parallel"}, clear=True),
            patch(
                "gpt_task.inference.tp.run_task_tp",
                return_value=runner_response,
            ) as run_task,
            patch(
                "gpt_task.inference.get_execution_dtype",
                return_value="float32",
            ),
        ):
            result = runner.inference(
                task_type=TaskType.LLM,
                task_args=LLM_TASK_ARGS,
                execution_model_files=LLM_MODEL_FILES,
                model_cache=MagicMock(),
                sd_config=MagicMock(),
                gpt_config=MagicMock(),
                output_dir=output_dir,
            )

        self.assertEqual(result.execution_dtype, "float32")
        run_task.assert_called_once()
        self.assertEqual(run_task.call_args.args[0].revision, SHA)

    def test_gpt_runner_accepts_legacy_api_without_dtype_getter(self):
        runner = HFTaskRunner()

        with (
            tempfile.TemporaryDirectory() as output_dir,
            patch.dict(os.environ, {}, clear=True),
            patch("gpt_task.inference.run_task", return_value={"choices": []}),
            patch("gpt_task.inference.get_execution_dtype", new=None),
        ):
            result = runner.inference(
                task_type=TaskType.LLM,
                task_args=LLM_TASK_ARGS,
                execution_model_files=LLM_MODEL_FILES,
                model_cache=MagicMock(),
                sd_config=MagicMock(),
                gpt_config=MagicMock(),
                output_dir=output_dir,
            )

        self.assertIsNone(result.execution_dtype)

    def test_wire_protocol_omits_missing_dtype(self):
        result_queue = Queue()
        result_queue.put(
            TaskResult(
                task_name="inference",
                task_id_commitment="task",
                result=SuccessResult(status="success"),
            )
        )
        worker = TaskWorker.__new__(TaskWorker)
        worker._status = "running"
        ws = MagicMock()
        ws.send.side_effect = lambda _: setattr(worker, "_status", "stopped")

        worker.result_consumer(ws, result_queue)

        payload = json.loads(ws.send.call_args.args[0])
        self.assertEqual(payload["result"], {"status": "success"})

    def test_wire_protocol_includes_reported_dtype(self):
        result = TaskResult(
            task_name="inference",
            task_id_commitment="task",
            result=SuccessResult(
                status="success",
                execution_dtype="float16",
            ),
        )

        payload = json.loads(result.model_dump_json(exclude_none=True))

        self.assertEqual(payload["result"]["execution_dtype"], "float16")


if __name__ == "__main__":
    unittest.main()
