import logging
import os
import signal
import traceback
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO, TextIOBase
from multiprocessing.connection import Connection
from typing import Type

from gpt_task.config import Config as GPTConfig
from sd_task.config import Config as SDConfig

from crynux_worker.config import Config
from crynux_worker.model.task import DownloadTaskInput, DownloadTaskResult

from .runner import TaskRunner

_logger = logging.getLogger(__name__)


class TeeOut(StringIO):
    def __init__(self, pipe: Connection, file: TextIOBase):
        self.pipe = pipe
        self.file = file

    def write(self, s: str) -> int:
        self.pipe.send(s.strip())
        self.file.write(s)
        return len(s)

    def isatty(self):
        return False


def download_process(
    pipe: Connection,
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
                        if pipe.poll(0):
                            task_input: DownloadTaskInput = pipe.recv()
                            try:
                                task_runner.download_model(
                                    task_type=task_input.task_type,
                                    model_type=task_input.model_type,
                                    model_name=task_input.model.id,
                                    variant=task_input.model.variant,
                                    sd_config=sd_config,
                                    gpt_config=gpt_config,
                                )
                                res = DownloadTaskResult(
                                    status="success", task_input=task_input
                                )
                            except Exception:
                                tb = traceback.format_exc()
                                res = DownloadTaskResult(
                                    status="error", traceback=tb, task_input=task_input
                                )
                                pipe.send(res)

                    except (EOFError, BrokenPipeError):
                        break

    except KeyboardInterrupt:
        pass
