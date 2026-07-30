"""GPU-resident model ownership transitions for the inference worker.

The inference worker MUST keep exactly one GPU model owner at a time:
the worker-level ModelCache (SD / classic LLM) or the gpt-task TP rank
group. Cross-backend transitions MUST evict the previous owner before the
next backend loads. Same-backend reuse remains allowed.
"""

import os

from crynux_worker.model import TaskType
from crynux_worker.model_cache import ModelCache


def prepare_gpu_for_task(task_type: TaskType, model_cache: ModelCache) -> None:
    """Evict conflicting GPU-resident caches before dispatching a task."""
    if (
        task_type == TaskType.LLM
        and os.environ.get("GPT_EXECUTOR") == "tensor_parallel"
    ):
        # run_task_tp owns the TP eligibility decision: eligible tasks clear
        # the worker cache, while classic fallback shuts down the rank group.
        return

    from gpt_task.inference import shutdown_tp_executor

    shutdown_tp_executor()
    if task_type == TaskType.SD_FT_LORA:
        # Fine-tuning loads outside ModelCache; drop any cached SD/classic
        # weights so they cannot share VRAM with the training load.
        model_cache.clear()


def release_gpu_residency(model_cache: ModelCache) -> None:
    """Release every GPU-resident model owner on inference process exit."""
    from gpt_task.inference import shutdown_tp_executor

    shutdown_tp_executor()
    model_cache.clear()
