import logging
import socket
from datetime import datetime
from multiprocessing import get_context
from selectors import EVENT_READ, DefaultSelector
from threading import Thread
from typing import Literal, Type

from gpt_task.config import Config as GPTConfig
from sd_task.config import Config as SDConfig
from websockets.sync.connection import Connection as WSConnection

from crynux_worker.config import Config
from crynux_worker.model import (DownloadTaskInput, DownloadTaskResult,
                                 InferenceTaskInput, InferenceTaskResult,
                                 TaskErrorResponse, TaskInput,
                                 TaskSuccessResponse)

from .download import download_process
from .inference import inference_process
from .runner import TaskRunner

_logger = logging.getLogger(__name__)

TaskStatus = Literal["running", "cancelled", "stopped"]


class TaskWorker(object):
    def __init__(
        self,
        task_runner_cls: Type[TaskRunner],
        config: Config,
        sd_config: SDConfig,
        gpt_config: GPTConfig,
    ) -> None:
        self._config = config
        self._sd_config = sd_config
        self._gpt_config = gpt_config

        ctx = get_context("spawn")

        inference_parent_pipe, inference_child_pipe = ctx.Pipe()
        self._inference_parent_pipe = inference_parent_pipe
        self._inference_child_pipe = inference_child_pipe
        self._inference_process = ctx.Process(
            target=inference_process,
            args=(inference_child_pipe, task_runner_cls, config, sd_config, gpt_config),
        )

        download_parent_pipe, download_child_pipe = ctx.Pipe()
        self._download_parent_pipe = download_parent_pipe
        self._download_child_pipe = download_child_pipe
        self._download_process = ctx.Process(
            target=download_process,
            args=(download_child_pipe, task_runner_cls, config, sd_config, gpt_config),
        )

        interrupt_read, interrupt_write = socket.socketpair()
        self._interrupt_read = interrupt_read
        self._interrupt_write = interrupt_write

        self._status: TaskStatus = "stopped"

        self._selector = DefaultSelector()
        self._selector.register(self._interrupt_read, EVENT_READ)

    def cancel(self):
        if self._status != "cancelled" and self._status != "stopped":
            _logger.info("cancel inference task")
            self._status = "cancelled"
            self._interrupt_write.send(b"\0")
            self._interrupt_write.close()

    def consumer(self, ws: WSConnection):
        while self._status == "running":
            try:
                message = ws.recv(0.1)
                assert isinstance(message, str)
                if len(message) > 0:
                    task_input = TaskInput.model_validate_json(message)

                    if task_input.task_name == "inference":
                        task = task_input.task
                        assert isinstance(task, InferenceTaskInput)
                        _logger.info(
                            f"Inference task {task.task_id_commitment} starts at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        self._inference_parent_pipe.send(task)
                    elif task_input.task_name == "download":
                        task = task_input.task
                        assert isinstance(task, DownloadTaskInput)
                        _logger.info(
                            f"Download task {task.task_id_commitment} model {task.model.id} starts at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        self._download_parent_pipe.send(task)
            except TimeoutError:
                pass

    def producer(self, ws: WSConnection):
        while self._status == "running":
            if self._inference_parent_pipe.poll(0.1):
                res1: InferenceTaskResult = self._inference_parent_pipe.recv()
                if res1.status == "success":
                    resp = TaskSuccessResponse(
                        task_name="inference",
                        task_id_commitment=res1.task_input.task_id_commitment,
                    )
                    ws.send(resp.model_dump_json())
                elif res1.status == "error":
                    assert res1.traceback is not None
                    resp = TaskErrorResponse(
                        task_name="inference",
                        task_id_commitment=res1.task_input.task_id_commitment,
                        traceback=res1.traceback,
                    )
                    ws.send(resp.model_dump_json())
            if self._download_parent_pipe.poll(0.1):
                res2: DownloadTaskResult = self._download_parent_pipe.recv()
                if res2.status == "success":
                    resp = TaskSuccessResponse(
                        task_name="download",
                        task_id_commitment=res2.task_input.task_id_commitment,
                    )
                    ws.send(resp.model_dump_json())
                elif res2.status == "error":
                    assert res2.traceback is not None
                    resp = TaskErrorResponse(
                        task_name="download",
                        task_id_commitment=res2.task_input.task_id_commitment,
                        traceback=res2.traceback,
                    )
                    ws.send(resp.model_dump_json())

    def run(self, ws: WSConnection):
        if self._status == "cancelled":
            return
        assert self._status == "stopped"
        self._inference_process.start()
        self._download_process.start()
        self._status = "running"

        consumer_thread = Thread(target=self.consumer, args=(ws,))
        consumer_thread.start()

        producer_thread = Thread(target=self.producer, args=(ws,))
        producer_thread.start()

        stop = False

        try:
            while not stop:
                for key, _ in self._selector.select(0.1):
                    if key.fileobj == self._interrupt_read:
                        self._inference_process.terminate()
                        _logger.info("inference task is cancelled")
                        self._download_process.kill()
                        _logger.info("download task is cancelled")
                        self._selector.unregister(self._interrupt_read)
                        self._interrupt_read.close()
                        stop = True
        except Exception as e:
            _logger.error("Worker unexpectederror")
            _logger.exception(e)
            self._inference_process.kill()
            _logger.info("inference task is killed")
            self._download_process.kill()
            _logger.info("download task is killed")
        finally:
            self._status = "stopped"
            producer_thread.join()
            consumer_thread.join()

            self._inference_process.join()
            _logger.info("inference task process is joined")
            self._download_process.join()
            _logger.info("download task process is joined")

            self._inference_child_pipe.close()
            self._inference_parent_pipe.close()

            self._download_child_pipe.close()
            self._download_parent_pipe.close()

            self._selector.close()
