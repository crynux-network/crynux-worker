import logging
import os
import signal
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from queue import Empty, Queue
from typing import Type

from gpt_task.config import Config as GPTConfig
from pydantic import ValidationError
from sd_task.config import Config as SDConfig

from crynux_worker.config import Config
from crynux_worker.model import (
    ErrorResult,
    InferenceTaskInput,
    SuccessResult,
    TaskResult,
)
from crynux_worker.model_cache import ModelCache

from .model_mutex import ModelMutex
from .runner import TaskRunner

_logger = logging.getLogger(__name__)


def _inference_one_task(
    task_runner: TaskRunner,
    task_input: InferenceTaskInput,
    model_cache: ModelCache,
    sd_config: SDConfig,
    gpt_config: GPTConfig,
    output_dir: str,
):

    try:
        results = task_runner.inference(
            task_type=task_input.task_type,
            task_args=task_input.task_args,
            model_cache=model_cache,
            sd_config=sd_config,
            gpt_config=gpt_config,
            output_dir=output_dir,
        )

        return results
    except ValidationError as e:
        raise ValueError("Task args invalid") from e


def inference_worker(
    task_input_queue: Queue[InferenceTaskInput],
    result_queue: Queue[TaskResult],
    model_mutex: ModelMutex,
    task_runner_cls: Type[TaskRunner],
    config: Config,
    sd_config: SDConfig,
    gpt_config: GPTConfig,
):
    try:
        task_runner = task_runner_cls()
        inference_log_file = os.path.join(config.log.dir, "crynux_worker_inference.log")
        with open(inference_log_file, mode="a", encoding="utf-8") as f:
            with redirect_stderr(f), redirect_stdout(f):
                logging.basicConfig(
                    format="[{asctime}] [{levelname:<8}] {name}: {message}",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    style="{",
                    level=logging.INFO,
                )
                model_cache = ModelCache()

                stop = False

                def _signal_handler(*args):
                    nonlocal stop
                    stop = True
                    logging.info("try to stop inference process gracefully")

                signal.signal(signal.SIGTERM, _signal_handler)

                while not stop:
                    try:
                        task_input: InferenceTaskInput = task_input_queue.get(
                            timeout=0.1
                        )
                        model_ids = [model.to_model_id() for model in task_input.models]
                        with model_mutex.lock_models(model_ids):
                            try:
                                _logger.info(
                                    f"Inference task {task_input.task_id} starts at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                                )
                                _inference_one_task(
                                    task_runner=task_runner,
                                    task_input=task_input,
                                    model_cache=model_cache,
                                    sd_config=sd_config,
                                    gpt_config=gpt_config,
                                    output_dir=task_input.output_dir,
                                )
                                res = TaskResult(
                                    task_name="inference",
                                    task_id_commitment=task_input.task_id,
                                    result=SuccessResult(status="success"),
                                )
                            except Exception as e:
                                _logger.exception(e)
                                tb = traceback.format_exc()
                                res = TaskResult(
                                    task_name="inference",
                                    task_id_commitment=task_input.task_id,
                                    result=ErrorResult(status="error", traceback=tb),
                                )

                        result_queue.put(res)
                    except Empty:
                        pass
                _logger.info("inference process exit normally")
    except KeyboardInterrupt:
        pass
