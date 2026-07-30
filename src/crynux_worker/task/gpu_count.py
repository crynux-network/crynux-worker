"""Resolve how many GPUs a failed worker task actually used."""

from __future__ import annotations

import logging

from crynux_worker.model import TaskType

_logger = logging.getLogger(__name__)


def visible_gpu_count() -> int:
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        _logger.exception("Failed to read visible CUDA device count")
        return 0


def executed_gpu_count_for_inference(task_type: TaskType) -> int:
    """Return GPUs used by the failed inference task.

    GPT tasks prefer the count recorded by gpt-task (TP world size or classic
    visible count). SD and other non-TP paths report the visible GPU count.
    When gpt-task never recorded a count, fall back to the visible count.
    """
    if task_type == TaskType.LLM:
        try:
            from gpt_task.inference import get_executed_gpu_count

            count = get_executed_gpu_count()
            if count is not None and count > 0:
                return count
        except Exception:
            _logger.exception("Failed to read gpt-task executed GPU count")
    return visible_gpu_count()
