import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
import shutil
import tempfile
from typing import Dict, Literal, Sequence

from gpt_task.config import Config as GPTConfig
from sd_task.config import Config as SDConfig

from crynux_worker.model import ModelFileEntry, TaskType
from crynux_worker.model.model_file import (
    IMAGE_BASE_MODEL_FORMAT,
    SOURCE_TYPE_HUGGINGFACE,
)
from crynux_worker.model_cache import ModelCache

_logger = logging.getLogger(__name__)

# Revision reported by the mock runner for sources without a revision
MOCK_REVISION = "0" * 40


@dataclass(frozen=True)
class InferenceExecutionResult:
    execution_dtype: str | None


def _resolve_revision(model_file: ModelFileEntry) -> str:
    """Return the concrete commit SHA the entry downloads."""
    if model_file.is_exact:
        return model_file.revision

    from huggingface_hub import model_info

    sha = model_info(model_file.repository).sha
    if not sha:
        raise ValueError(f"No commit found for model {model_file.repository}")
    return sha


def _repo_has_model_index(repository: str, revision: str) -> bool:
    from huggingface_hub import list_repo_files

    files = list_repo_files(repository, revision=revision)
    return any(
        name == "model_index.json" or name.endswith("/model_index.json")
        for name in files
    )


class TaskRunner(ABC):
    @abstractmethod
    def download_model(
        self,
        task_type: TaskType,
        model_type: Literal["base", "lora", "controlnet"],
        model_name: str,
        variant: str | None,
        sd_config: SDConfig,
        gpt_config: GPTConfig,
    ): ...

    @abstractmethod
    def download_model_file(
        self,
        task_type: TaskType,
        model_file: ModelFileEntry,
        sd_config: SDConfig,
        gpt_config: GPTConfig,
    ) -> ModelFileEntry:
        """Download the model file and return its entry with the concrete revision."""

    @abstractmethod
    def inference(
        self,
        task_type: TaskType,
        task_args: str,
        execution_model_files: Sequence[ModelFileEntry],
        model_cache: ModelCache,
        sd_config: SDConfig,
        gpt_config: GPTConfig,
        output_dir: str,
    ): ...


class HFTaskRunner(TaskRunner):
    def download_model_file(
        self,
        task_type: TaskType,
        model_file: ModelFileEntry,
        sd_config: SDConfig,
        gpt_config: GPTConfig,
    ) -> ModelFileEntry:
        if model_file.source_type != SOURCE_TYPE_HUGGINGFACE:
            raise ValueError(
                f"Unsupported model source type: {model_file.source_type}"
            )

        if task_type == TaskType.LLM:
            from gpt_task.prefetch import download_model, requests_proxy_session

            if model_file.format != "":
                raise ValueError(
                    f"Unsupported gpt model file format: {model_file.format}"
                )
            with requests_proxy_session(gpt_config.proxy):
                revision = _resolve_revision(model_file)
            if gpt_config.data_dir is None:
                hf_cache_dir = None
            else:
                hf_cache_dir = gpt_config.data_dir.models.huggingface
            download_model(
                model_file.repository, hf_cache_dir, gpt_config.proxy, revision
            )
        else:
            from sd_task.download_model import (
                check_and_download_hf_pipeline,
                requests_proxy_session,
            )
            from sd_task.task_args.image_task import ComponentSource
            from sd_task.task_runner.image_task.download_model import (
                download_component_file,
            )

            hf_cache_dir = sd_config.data_dir.models.huggingface
            with requests_proxy_session(sd_config.proxy):
                revision = _resolve_revision(model_file)
                is_pipeline = model_file.format == IMAGE_BASE_MODEL_FORMAT and (
                    _repo_has_model_index(model_file.repository, revision)
                )
            if is_pipeline:
                check_and_download_hf_pipeline(
                    model_file.repository,
                    revision=revision,
                    hf_model_cache_dir=hf_cache_dir,
                    proxy=sd_config.proxy,
                )
            elif model_file.format:
                download_component_file(
                    ComponentSource(
                        format=model_file.format,
                        repo=model_file.repository,
                        revision=revision,
                        format_kwargs=model_file.format_kwargs,
                    ),
                    hf_cache_dir,
                    sd_config.proxy,
                )
            else:
                raise ValueError(
                    f"Unsupported image model file format: {model_file.format}"
                )

        _logger.info(
            f"Successfully download {model_file.repository} at revision {revision}"
        )
        return model_file.model_copy(update={"revision": revision})

    def download_model(
        self,
        task_type: TaskType,
        model_type: Literal["base", "lora", "controlnet"],
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
            _logger.info(f"Successfully download gpt base model: {model_name}")
        else:
            from diffusers import ControlNetModel
            from diffusers.utils import SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME
            from sd_task.download_model import (
                check_and_download_hf_model, check_and_download_hf_pipeline,
                check_and_download_model_by_name)

            if model_type == "base":
                check_and_download_hf_pipeline(
                    model_name,
                    variant=variant,
                    hf_model_cache_dir=sd_config.data_dir.models.huggingface,
                    proxy=sd_config.proxy,
                )
                _logger.info(f"Successfully download sd base model: {model_name}")
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
                _logger.info(f"Successfully download sd controlnet model: {model_name}")
            elif model_type == "lora":
                check_and_download_model_by_name(
                    model_name,
                    None,
                    [],
                    True,
                    hf_model_cache_dir=sd_config.data_dir.models.huggingface,
                    external_model_cache_dir=sd_config.data_dir.models.external,
                    proxy=sd_config.proxy,
                    variant=variant,
                )
                _logger.info(f"Successfully download sd lora model: {model_name}")

    def inference(
        self,
        task_type: TaskType,
        task_args: str,
        execution_model_files: Sequence[ModelFileEntry],
        model_cache: ModelCache,
        sd_config: SDConfig,
        gpt_config: GPTConfig,
        output_dir: str,
    ):
        if task_type == TaskType.SD:
            from sd_task.task_args.image_task import ImageTaskArgs
            from sd_task.task_runner.image_task import (
                apply_execution_model_files,
                run_image_task,
            )

            args = ImageTaskArgs.model_validate_json(task_args)
            args = apply_execution_model_files(args, execution_model_files)
            imgs = run_image_task(args, model_cache=model_cache, config=sd_config)
            for i, img in enumerate(imgs):
                filename = os.path.join(output_dir, f"{i}.png")
                img.save(filename)
            return imgs
        elif task_type == TaskType.LLM:
            import gpt_task.inference as gpt_inference
            from gpt_task.inference import apply_execution_model_files
            from gpt_task.models import GPTTaskArgs

            args = GPTTaskArgs.model_validate_json(task_args)
            args = apply_execution_model_files(args, execution_model_files)
            if os.environ.get("GPT_EXECUTOR") == "tensor_parallel":
                # TP tasks cache shards inside the rank processes; the
                # worker-level model cache serves classic-fallback tasks
                # and is evicted by run_task_tp before a TP task runs.
                from gpt_task.inference.tp import run_task_tp

                resp = run_task_tp(args, model_cache=model_cache, config=gpt_config)
            else:
                from gpt_task.inference import run_task as run_gpt_task

                resp = run_gpt_task(
                    args, model_cache=model_cache, config=gpt_config
                )
            with open(
                os.path.join(output_dir, "0.json"), mode="w", encoding="utf-8"
            ) as f:
                json.dump(resp, f)
            get_execution_dtype = getattr(
                gpt_inference, "get_execution_dtype", None
            )
            execution_dtype = (
                get_execution_dtype() if get_execution_dtype is not None else None
            )
            return InferenceExecutionResult(execution_dtype=execution_dtype)
        elif task_type == TaskType.SD_FT_LORA:
            from sd_task.task_args import FinetuneLoraTaskArgs
            from sd_task.task_runner.finetune_task import \
                run_finetune_lora_task

            args = FinetuneLoraTaskArgs.model_validate_json(task_args)
            run_finetune_lora_task(args, output_dir=output_dir, config=sd_config)


class MockTaskRunner(TaskRunner):
    def download_model(
        self,
        task_type: TaskType,
        model_type: Literal["base", "lora", "controlnet"],
        model_name: str,
        variant: str | None,
        sd_config: SDConfig,
        gpt_config: GPTConfig,
    ):
        _logger.info(
            f"Successfully download {task_type} {model_type} model: {model_name}"
        )

    def download_model_file(
        self,
        task_type: TaskType,
        model_file: ModelFileEntry,
        sd_config: SDConfig,
        gpt_config: GPTConfig,
    ) -> ModelFileEntry:
        _logger.info(
            f"Successfully download {task_type} model file: {model_file.repository}"
        )
        if model_file.is_exact:
            return model_file
        return model_file.model_copy(update={"revision": MOCK_REVISION})

    def inference(
        self,
        task_type: TaskType,
        task_args: str,
        execution_model_files: Sequence[ModelFileEntry],
        model_cache: ModelCache,
        sd_config: SDConfig,
        gpt_config: GPTConfig,
        output_dir: str,
    ):
        if task_type == TaskType.SD:
            content = bytes.fromhex(
                "89504e470d0a1a0a0000000d49484452000000030000000308000000007343ea630000001449444154789c63f8cff09f89e13f0303c37f06001c0003ff136fd2d30000000049454e44ae426082"
            )
            with open(os.path.join(output_dir, "0.png"), mode="wb") as f:
                f.write(content)
        elif task_type == TaskType.LLM:
            resp = {
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
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 30,
                    "total_tokens": 41,
                },
            }
            with open(
                os.path.join(output_dir, "0.json"), mode="w", encoding="utf-8"
            ) as f:
                json.dump(resp, f)
        elif task_type == TaskType.SD_FT_LORA:
            from sd_task.task_args import FinetuneLoraTaskArgs
            args = FinetuneLoraTaskArgs.model_validate_json(task_args)
            if args.checkpoint is not None:
                assert os.path.exists(args.checkpoint)
                assert os.path.isdir(args.checkpoint)
            validation_dir = os.path.join(output_dir, "validation")
            checkpoint_dir = os.path.join(output_dir, "checkpoint")
            os.makedirs(validation_dir, exist_ok=True)
            os.makedirs(checkpoint_dir, exist_ok=True)
            with open(os.path.join(validation_dir, "0.png"), mode="wb") as f:
                content = bytes.fromhex(
                    "89504e470d0a1a0a0000000d49484452000000030000000308000000007343ea630000001449444154789c63f8cff09f89e13f0303c37f06001c0003ff136fd2d30000000049454e44ae426082"
                )
                f.write(content)

            global_step = 0
            global_epoch = 0
            finish = False
            if args.checkpoint is not None:
                if os.path.exists(os.path.join(args.checkpoint, "global_step.txt")):
                    with open(os.path.join(args.checkpoint, "global_step.txt"), mode="r", encoding="utf-8") as f:
                        global_step = int(f.read().strip())
                if os.path.exists(os.path.join(args.checkpoint, "global_epoch.txt")):
                    with open(os.path.join(args.checkpoint, "global_epoch.txt"), mode="r", encoding="utf-8") as f:
                        global_epoch = int(f.read().strip())
            if args.train_args.max_train_steps is not None and args.train_args.num_train_steps is not None and args.train_args.max_train_steps > 0:
                global_step += args.train_args.num_train_steps
                with open(os.path.join(checkpoint_dir, "global_step.txt"), mode="w", encoding="utf-8") as f:
                    f.write(str(global_step))
                if global_step >= args.train_args.max_train_steps:
                    finish = True
            elif args.train_args.max_train_epochs > 0 and args.train_args.num_train_epochs > 0:
                global_epoch += args.train_args.num_train_epochs
                with open(os.path.join(checkpoint_dir, "global_epoch.txt"), mode="w", encoding="utf-8") as f:
                    f.write(str(global_epoch))
                if global_epoch >= args.train_args.max_train_epochs:
                    finish = True  
            else:
                raise ValueError("max_train_steps or max_train_epochs must be set")

            if finish:
                with open(os.path.join(checkpoint_dir, "FINISH"), mode="w", encoding="utf-8") as f:
                    f.write("")
