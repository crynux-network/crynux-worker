import logging
import os
import signal
import traceback
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO, TextIOBase
from multiprocessing.connection import Connection
from queue import Empty, Queue
from typing import Type

from gpt_task.config import Config as GPTConfig
from sd_task.config import Config as SDConfig

from crynux_worker.config import Config
from crynux_worker.model.task import DownloadTaskInput, TaskResult

from .runner import TaskRunner
from .model_mutex import ModelMutex

_logger = logging.getLogger(__name__)


def download_worker(
    task_input_queue: Queue[DownloadTaskInput],
    result_queue: Queue[TaskResult],
    model_mutex: ModelMutex,
    task_runner_cls: Type[TaskRunner],
    config: Config,
    sd_config: SDConfig,
    gpt_config: GPTConfig,
):
    task_runner = task_runner_cls()
    try:
        prefetch_log_file = os.path.join(config.log.dir, "crynux_worker_prefetch.log")
        with open(prefetch_log_file, mode="a", encoding="utf-8") as f:
            with redirect_stderr(f), redirect_stdout(f):
                logging.basicConfig(
                    format="[{asctime}] [{levelname:<8}] {name}: {message}",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    style="{",
                    level=logging.INFO,
                )

                stop = False

                def _signal_handler(*args):
                    nonlocal stop
                    stop = True
                    logging.info("try to stop inference process gracefully")

                signal.signal(signal.SIGTERM, _signal_handler)

                while not stop:
                    try:
                        task_input = task_input_queue.get(timeout=0.1)
                        model_id = task_input.model.id
                        with model_mutex.lock(model_id):
                            try:
                                task_runner.download_model(
                                    task_type=task_input.task_type,
                                    model_type=task_input.model_type,
                                    model_name=task_input.model.id,
                                    variant=task_input.model.variant,
                                    sd_config=sd_config,
                                    gpt_config=gpt_config,
                                )
                                res = TaskResult(
                                    task_name="download",
                                    status="success",
                                    task_id_commitment=task_input.task_id_commitment,
                                )
                            except Exception:
                                tb = traceback.format_exc()
                                res = TaskResult(
                                    task_name="download",
                                    status="error",
                                    task_id_commitment=task_input.task_id_commitment,
                                    traceback=tb,
                                )
                        result_queue.put(res)

                    except Empty:
                        pass
                _logger.info("download process exit normally")

    except KeyboardInterrupt:
        pass
