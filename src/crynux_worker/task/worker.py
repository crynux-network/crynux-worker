import concurrent.futures
import logging
from multiprocessing import get_context
from queue import Empty, Queue
from typing import Literal, Type

from gpt_task.config import Config as GPTConfig
from sd_task.config import Config as SDConfig
from websockets.sync.connection import Connection as WSConnection

from crynux_worker.config import Config
from crynux_worker.model import (DownloadTaskInput, InferenceTaskInput,
                                 TaskInput, TaskResult)

from .download import download_worker
from .inference import inference_worker
from .model_mutex import ModelMutex
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
        self._task_runner_cls = task_runner_cls
        self._config = config
        self._sd_config = sd_config
        self._gpt_config = gpt_config

        self._mp_ctx = get_context("spawn")
        self._mp_manager = self._mp_ctx.Manager()
        self._inference_task_queue: Queue[InferenceTaskInput] = self._mp_manager.Queue()
        self._download_task_queue: Queue[DownloadTaskInput] = self._mp_manager.Queue()
        self._result_queue: Queue[TaskResult] = self._mp_manager.Queue()
        self._model_mutex = ModelMutex(self._mp_manager)

        self._condition = self._mp_manager.Condition()
        self._using_models = self._mp_manager.dict()

        self._status: TaskStatus = "stopped"

    def cancel(self):
        if self._status == "running":
            _logger.info("cancel inference task")
            self._status = "cancelled"

    def task_producer(self, ws: WSConnection):
        while self._status == "running":
            try:
                message = ws.recv(0.1)
                assert isinstance(message, str)
                if len(message) > 0:
                    task_input = TaskInput.model_validate_json(message)

                    if task_input.task.task_name == "inference":
                        task = task_input.task
                        self._inference_task_queue.put(task)
                    elif task_input.task.task_name == "download":
                        task = task_input.task
                        self._download_task_queue.put(task)
            except TimeoutError:
                pass

    def result_consumer(self, ws: WSConnection):
        while self._status == "running":
            try:
                res = self._result_queue.get(timeout=0.1)
                ws.send(res.model_dump_json())
            except Empty:
                pass

    def run(self, ws: WSConnection):
        if self._status == "cancelled":
            return
        assert self._status == "stopped"

        self._mp_manager.start()

        inference_process = self._mp_ctx.Process(
            target=inference_worker,
            args=(
                self._inference_task_queue,
                self._result_queue,
                self._model_mutex,
                self._task_runner_cls,
                self._config,
                self._sd_config,
                self._gpt_config,
            ),
        )
        inference_process.start()
        download_process = self._mp_ctx.Process(
            target=download_worker,
            args=(
                self._download_task_queue,
                self._result_queue,
                self._model_mutex,
                self._task_runner_cls,
                self._config,
                self._sd_config,
                self._gpt_config,
            ),
        )
        download_process.start()

        self._status = "running"
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        try:
            task_producer_fut = pool.submit(self.task_producer, ws)
            result_consumer_fut = pool.submit(self.result_consumer, ws)
            done, _ = concurrent.futures.wait(
                [task_producer_fut, result_consumer_fut],
                return_when=concurrent.futures.FIRST_EXCEPTION,
            )
            has_error = False
            for fut in done:
                exc = fut.exception()
                if exc is not None:
                    _logger.error("Worker running error")
                    _logger.exception(exc)
                    has_error = True

            if has_error:
                if inference_process.is_alive():
                    inference_process.kill()
                    _logger.info("close inference task forcely")
                if download_process.is_alive():
                    download_process.kill()
                    _logger.info("close download task forcely")
            else:
                if inference_process.is_alive():
                    inference_process.terminate()
                    _logger.info("close inference task gracefully")
                if download_process.is_alive():
                    download_process.terminate()
                    _logger.info("close download task gracefully")

        except Exception as e:
            _logger.error("Worker unexpected error")
            _logger.exception(e)
            if inference_process.is_alive():
                inference_process.kill()
                _logger.info("close inference task forcely")
            if download_process.is_alive():
                download_process.kill()
                _logger.info("close inference task forcely")
        finally:
            self._status = "stopped"
            pool.shutdown(wait=True, cancel_futures=True)
            inference_process.join()
            _logger.info("inference task process is joined")
            download_process.join()
            _logger.info("download task process is joined")

            self._mp_manager.shutdown()
