import json
import os
from abc import ABC, abstractmethod
from typing import Literal

from gpt_task.config import Config as GPTConfig
from sd_task.config import Config as SDConfig

from crynux_worker.model import TaskType
from crynux_worker.model_cache import ModelCache


class TaskRunner(ABC):
    @abstractmethod
    def download_model(
        self,
        task_type: TaskType,
        model_type: Literal["base", "vae", "controlnet"],
        model_name: str,
        variant: str | None,
        sd_config: SDConfig,
        gpt_config: GPTConfig,
    ): ...

    @abstractmethod
    def inference(
        self,
        task_type: TaskType,
        task_args: str,
        model_cache: ModelCache,
        sd_config: SDConfig,
        gpt_config: GPTConfig,
        output_dir: str,
    ): ...


class HFTaskRunner(TaskRunner):
    def download_model(
        self,
        task_type: TaskType,
        model_type: Literal["base"] | Literal["vae"] | Literal["controlnet"],
        model_name: str,
        variant: str | None,
        sd_config: SDConfig,
        gpt_config: GPTConfig,
    ):
        if task_type == TaskType.LLM:
            from gpt_task.prefetch import download_model

            if gpt_config.data_dir is None:
                hf_cache_dir = None
            else:
                hf_cache_dir = gpt_config.data_dir.models.huggingface

            download_model(model_name, hf_cache_dir, gpt_config.proxy)
        else:
            from diffusers import AutoencoderKL, ControlNetModel
            from diffusers.utils import SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME
            from sd_task.download_model import (check_and_download_hf_model,
                                                check_and_download_hf_pipeline)

            if model_type == "base":
                check_and_download_hf_pipeline(
                    model_name,
                    variant=variant,
                    hf_model_cache_dir=sd_config.data_dir.models.huggingface,
                    proxy=sd_config.proxy,
                )
            elif model_type == "controlnet":
                check_and_download_hf_model(
                    model_name,
                    ControlNetModel.load_config,
                    [SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME],
                    False,
                    sd_config.data_dir.models.huggingface,
                    sd_config.proxy,
                    variant,
                )
            elif model_type == "vae":
                check_and_download_hf_model(
                    model_name,
                    AutoencoderKL.load_config,
                    [SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME],
                    False,
                    sd_config.data_dir.models.huggingface,
                    sd_config.proxy,
                    variant,
                )

    def inference(
        self,
        task_type: TaskType,
        task_args: str,
        model_cache: ModelCache,
        sd_config: SDConfig,
        gpt_config: GPTConfig,
        output_dir: str,
    ):
        if task_type == TaskType.SD:
            from sd_task.task_args.inference_task import InferenceTaskArgs
            from sd_task.task_runner import run_inference_task

            args = InferenceTaskArgs.model_validate_json(task_args)
            imgs = run_inference_task(args, model_cache=model_cache, config=sd_config)
            for i, img in enumerate(imgs):
                filename = os.path.join(output_dir, f"{i}.png")
                img.save(filename)
        elif task_type == TaskType.LLM:
            from gpt_task.inference import run_task as run_gpt_task
            from gpt_task.models import GPTTaskArgs

            args = GPTTaskArgs.model_validate_json(task_args)
            resp = run_gpt_task(args, model_cache=model_cache, config=gpt_config)
            with open(
                os.path.join(output_dir, "0.json"), mode="w", encoding="utf-8"
            ) as f:
                json.dump(resp, f)
        elif task_type == TaskType.SD_FT_LORA:
            from sd_task.task_args import FinetuneLoraTaskArgs
            from sd_task.task_runner import run_finetune_lora_task

            args = FinetuneLoraTaskArgs.model_validate_json(task_args)
            run_finetune_lora_task(args, output_dir=output_dir, config=sd_config)


class MockTaskRunner(TaskRunner):
    def download_model(
        self,
        task_type: TaskType,
        model_type: Literal["base"] | Literal["vae"] | Literal["controlnet"],
        model_name: str,
        variant: str | None,
        sd_config: SDConfig,
        gpt_config: GPTConfig,
    ):
        pass

    def inference(
        self,
        task_type: TaskType,
        task_args: str,
        model_cache: ModelCache,
        sd_config: SDConfig,
        gpt_config: GPTConfig,
        output_dir: str,
    ):
        if task_type == TaskType.SD:
            content = bytes.fromhex(
                "89504e470d0a1a0a0000000d4948445200000008000000080800000000e164e1570000000c49444154789c6360a00e0000004800012eb83c7e0000000049454e44ae426082"
            )
            with open(os.path.join(output_dir, "0.png"), mode="wb") as f:
                f.write(content)
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
            with open(
                os.path.join(output_dir, "0.json"), mode="w", encoding="utf-8"
            ) as f:
                f.write(resp)
        elif task_type == TaskType.SD_FT_LORA:
            os.mkdir(os.path.join(output_dir, "validation"))
            os.mkdir(os.path.join(output_dir, "checkpoint"))
