import json
import os
from abc import ABC, abstractmethod
from io import BytesIO
from typing import List

from gpt_task.config import Config as GPTConfig
from sd_task.config import Config as SDConfig

from crynux_worker.model import TaskType
from crynux_worker.model_cache import ModelCache
from PIL.Image import Image


class TaskRunner(ABC):
    @abstractmethod
    def prefetch_models(self, sd_config: SDConfig, gpt_config: GPTConfig): ...

    @abstractmethod
    def run_inference_task(
        self,
        task_type: TaskType,
        task_args: str,
        model_cache: ModelCache,
        sd_config: SDConfig,
        gpt_config: GPTConfig,
        output_dir: str | None = None,
    ) -> List[str | bytes]: ...


class HFTaskRunner(TaskRunner):
    def prefetch_models(self, sd_config: SDConfig, gpt_config: GPTConfig):
        from gpt_task.prefetch import prefetch_models as gpt_prefetch_models
        from sd_task.prefetch import prefetch_models as sd_prefetch_models

        sd_prefetch_models(sd_config)
        gpt_prefetch_models(gpt_config)

    def run_inference_task(
        self,
        task_type: TaskType,
        task_args: str,
        model_cache: ModelCache,
        sd_config: SDConfig,
        gpt_config: GPTConfig,
        output_dir: str | None = None,
    ) -> List[str | bytes]:
        results: List[str | bytes] = []

        if task_type == TaskType.SD:
            from sd_task.task_args.inference_task import InferenceTaskArgs
            from sd_task.task_runner import run_inference_task

            args = InferenceTaskArgs.model_validate_json(task_args)
            imgs = run_inference_task(args, model_cache=model_cache, config=sd_config)
            for img in imgs:
                f = BytesIO()
                img.save(f, format="PNG")
                img_bytes = f.getvalue()
                results.append(img_bytes)
        elif task_type == TaskType.LLM:
            from gpt_task.inference import run_task as run_gpt_task
            from gpt_task.models import GPTTaskArgs

            args = GPTTaskArgs.model_validate_json(task_args)
            resp = run_gpt_task(args, model_cache=model_cache, config=gpt_config)
            resp_json_str = json.dumps(resp)
            results.append(resp_json_str)
        elif task_type == TaskType.SD_FT_LORA:
            from sd_task.task_args import FinetuneLoraTaskArgs
            from sd_task.task_runner import run_finetune_lora_task

            assert output_dir is not None

            args = FinetuneLoraTaskArgs.model_validate_json(task_args)
            run_finetune_lora_task(args, output_dir=output_dir, config=sd_config)
            results.append(os.path.abspath(output_dir))

        return results


class MockTaskRunner(TaskRunner):
    def prefetch_models(self, sd_config: SDConfig, gpt_config: GPTConfig):
        if sd_config.preloaded_models.base is not None:
            for model in sd_config.preloaded_models.base:
                print(f"Preloading base model: {model.id}")
                print(f"Successfully preloaded base model: {model.id}")
        if gpt_config.preloaded_models.base is not None:
            for model in gpt_config.preloaded_models.base:
                print(f"Preloading base model: {model.id}")
                print(f"Successfully preloaded base model: {model.id}")

    def run_inference_task(
        self,
        task_type: TaskType,
        task_args: str,
        model_cache: ModelCache,
        sd_config: SDConfig,
        gpt_config: GPTConfig,
        output_dir: str | None = None,
    ) -> List[str | bytes]:
        results: List[str | bytes] = []

        if task_type == TaskType.SD:
            results.append(
                bytes.fromhex(
                    "89504e470d0a1a0a0000000d4948445200000008000000080800000000e164e1570000000c49444154789c6360a00e0000004800012eb83c7e0000000049454e44ae426082"
                )
            )
        elif task_type == TaskType.LLM:
            resp = """
            {
                "model": "gpt2",
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "role": "assistant",
                            "content": '\n\nI have a chat bot, called "Eleanor" which was developed by my team on Skype. '
                            "The only thing I will say is this",
                        },
                        "index": 0,
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 30, "total_tokens": 41},
            }"""
            results.append(resp.strip())
        elif task_type == TaskType.SD_FT_LORA:
            assert output_dir is not None

            results.append(output_dir)

        return results
