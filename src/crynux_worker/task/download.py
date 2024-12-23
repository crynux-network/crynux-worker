import json
import logging
import os
import re
import signal
import socket
import traceback
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO, TextIOBase
from multiprocessing import get_context
from multiprocessing.connection import Connection
from selectors import EVENT_READ, DefaultSelector
from threading import Thread
from typing import Literal, Type

from gpt_task.config import Config as GPTConfig
from sd_task.config import Config as SDConfig
from websockets.sync.connection import Connection as WSConnection

from crynux_worker.config import Config
from crynux_worker.log import init as log_init
from crynux_worker.model import PayloadType, WorkerPayloadMessage, WorkerPhase
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


def _download_process(
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
                                    gpt_config=gpt_config
                                )
                                res = DownloadTaskResult(
                                    status="success",
                                    task_input=task_input
                                )
                            except Exception:
                                tb = traceback.format_exc()
                                res = DownloadTaskResult(
                                    status="error",
                                    traceback=tb,
                                    task_input=task_input
                                )
                                pipe.send(res)

                    except (EOFError, BrokenPipeError):
                        break

    except KeyboardInterrupt:
        pass


PrefetchStatus = Literal["idle", "running", "cancelled", "finished"]


class PrefetchTask(object):
    def __init__(
        self,
        task_runner_cls: Type[TaskRunner],
        config: Config,
        sd_config: SDConfig,
        gpt_config: GPTConfig,
        total_models: int,
    ):
        self.task_runner_cls = task_runner_cls
        self.config = config
        self.sd_config = sd_config
        self.gpt_config = gpt_config
        self.total_models = total_models

        interrupt_read, interrupt_write = socket.socketpair()
        self.interrupt_read = interrupt_read
        self.interrupt_write = interrupt_write

        self._status: PrefetchStatus = "idle"

    def cancel(self):
        if self._status != "cancelled" and self._status != "finished":
            _logger.info("cancel prefetch task")
            self._status = "cancelled"
            self.interrupt_write.send(b"\0")
            self.interrupt_write.close()

    def run(self, websocket: WSConnection):
        if self._status == "cancelled":
            return True

        phase_msg = {"phase": "prefetch"}
        websocket.send(json.dumps(phase_msg))

        _logger.info("Start downloading models")

        ctx = get_context("spawn")
        parent_pipe, child_pipe = ctx.Pipe()

        prefetch_process = ctx.Process(
            target=_download_process,
            args=(child_pipe, self.task_runner_cls, self.config, self.sd_config, self.gpt_config),
        )
        prefetch_process.start()

        def _monitor():
            prefetch_process.join()
            prefetch_process.close()
            child_pipe.close()

        monitor_thread = Thread(target=_monitor)
        monitor_thread.start()

        selector = DefaultSelector()
        selector.register(self.interrupt_read, EVENT_READ)

        current_models = 0
        pattern = re.compile(r"Preloading")

        try:
            while True:
                for key, _ in selector.select(0):
                    if key.fileobj == self.interrupt_read:
                        prefetch_process.kill()
                        _logger.info("prefetch task is cancelled")
                        selector.unregister(self.interrupt_read)
                        self.interrupt_read.close()
                        return True
                try:
                    if parent_pipe.poll(0):
                        content = parent_pipe.recv()
                        if isinstance(content, tuple):
                            assert content[0] == "error"
                            err_msg = content[1]
                            payload_msg = WorkerPayloadMessage(
                                worker_phase=WorkerPhase.Prefetch,
                                has_payload=True,
                                has_next=False,
                                payload_type=PayloadType.Error,
                            )
                            websocket.send(payload_msg.model_dump_json())
                            websocket.send(err_msg)

                            _logger.error("Downloading models failed")
                            _logger.error(err_msg)
                            return False
                        elif isinstance(content, str):
                            print(content)
                            if pattern.search(content) is not None:
                                current_models += 1
                                msg = f"Downloading models............ ({current_models}/{self.total_models})"
                                payload_msg = WorkerPayloadMessage(
                                    worker_phase=WorkerPhase.Prefetch,
                                    has_payload=True,
                                    has_next=True,
                                    payload_type=PayloadType.Text,
                                )
                                websocket.send(payload_msg.model_dump_json())
                                websocket.send(msg)
                except (EOFError, BrokenPipeError):
                    payload_msg = WorkerPayloadMessage(
                        worker_phase=WorkerPhase.Prefetch,
                        has_payload=False,
                        has_next=False,
                    )
                    websocket.send(payload_msg.model_dump_json())
                    return True
        finally:
            monitor_thread.join()
            parent_pipe.close()
            selector.close()
            self._status = "finished"
