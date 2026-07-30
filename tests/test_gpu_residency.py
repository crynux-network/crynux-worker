import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from crynux_worker.gpu_residency import prepare_gpu_for_task, release_gpu_residency
from crynux_worker.model import TaskType
from crynux_worker.task.inference import _inference_one_task


class GPUResidencyTests(unittest.TestCase):
    def test_sd_shuts_down_tp_without_clearing_worker_cache(self):
        model_cache = MagicMock()

        with patch(
            "gpt_task.inference.shutdown_tp_executor"
        ) as shutdown:
            prepare_gpu_for_task(TaskType.SD, model_cache)

        shutdown.assert_called_once()
        model_cache.clear.assert_not_called()

    def test_sd_finetune_shuts_down_tp_and_clears_worker_cache(self):
        model_cache = MagicMock()

        with patch(
            "gpt_task.inference.shutdown_tp_executor"
        ) as shutdown:
            prepare_gpu_for_task(TaskType.SD_FT_LORA, model_cache)

        shutdown.assert_called_once()
        model_cache.clear.assert_called_once()

    def test_tp_llm_leaves_transition_to_run_task_tp(self):
        model_cache = MagicMock()

        with (
            patch.dict("os.environ", {"GPT_EXECUTOR": "tensor_parallel"}),
            patch("gpt_task.inference.shutdown_tp_executor") as shutdown,
        ):
            prepare_gpu_for_task(TaskType.LLM, model_cache)

        shutdown.assert_not_called()
        model_cache.clear.assert_not_called()

    def test_classic_llm_shuts_down_tp(self):
        model_cache = MagicMock()

        with (
            patch.dict("os.environ", {}, clear=True),
            patch("gpt_task.inference.shutdown_tp_executor") as shutdown,
        ):
            prepare_gpu_for_task(TaskType.LLM, model_cache)

        shutdown.assert_called_once()
        model_cache.clear.assert_not_called()

    def test_inference_coordinator_prepares_before_runner(self):
        order = []
        model_cache = MagicMock()
        task_runner = MagicMock()
        task_runner.inference.side_effect = lambda **kwargs: order.append("run")
        task_input = SimpleNamespace(task_type=TaskType.SD, task_args="{}")

        with patch(
            "crynux_worker.task.inference.prepare_gpu_for_task",
            side_effect=lambda *args: order.append("prepare"),
        ) as prepare:
            _inference_one_task(
                task_runner=task_runner,
                task_input=task_input,
                model_cache=model_cache,
                sd_config=MagicMock(),
                gpt_config=MagicMock(),
                output_dir="output",
            )

        self.assertEqual(order, ["prepare", "run"])
        prepare.assert_called_once_with(TaskType.SD, model_cache)

    def test_release_clears_tp_and_worker_cache(self):
        model_cache = MagicMock()

        with patch(
            "gpt_task.inference.shutdown_tp_executor"
        ) as shutdown:
            release_gpu_residency(model_cache)

        shutdown.assert_called_once()
        model_cache.clear.assert_called_once()


if __name__ == "__main__":
    unittest.main()
