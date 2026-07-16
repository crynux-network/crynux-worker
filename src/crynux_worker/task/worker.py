import concurrent.futures
import logging
from multiprocessing import get_context
from queue import Empty, Queue
from typing import Literal, Type

from gpt_task.config import Config as GPTConfig
from sd_task.config import Config as SDConfig
from websockets.sync.connection import Connection as WSConnection

from crynux_worker.config import Config
from crynux_worker.model import TaskInput, TaskResult

from .download import download_worker
from .inference import inference_worker
from .runner import TaskRunner

_logger = logging.getLogger(__name__)

TaskStatus = Literal["running", "cancelled", "stopped"]


class TaskWorkerRunningError(Exception):
    def __str__(self):
        return "Task worker running error"


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

        self._status: TaskStatus = "stopped"

    def cancel(self):
        if self._status == "running":
            _logger.info("cancel task worker")
            self._status = "cancelled"

    def task_producer(self, ws: WSConnection, task_queue: Queue):
        task_name = (
            "inference" if self._config.worker_role == "inference" else "download"
        )
        while self._status == "running":
            try:
                message = ws.recv(0.1)
                assert isinstance(message, str)
                if len(message) > 0:
                    task_input = TaskInput.model_validate_json(message)

                    if task_input.task.task_name == task_name:
                        task_queue.put(task_input.task)
                    else:
                        _logger.warning(
                            f"Drop task {task_input.task.task_id}: task name "
                            f"{task_input.task.task_name} does not match the "
                            f"worker role {self._config.worker_role}"
                        )
            except TimeoutError:
                pass
            except Exception as e:
                _logger.error("task producer running error")
                _logger.exception(e)
                raise e

    def result_consumer(self, ws: WSConnection, result_queue: Queue[TaskResult]):
        while self._status == "running":
            try:
                res = result_queue.get(timeout=0.1)
                ws.send(res.model_dump_json())
            except Empty:
                pass
            except Exception as e:
                _logger.error("result consumer running error")
                _logger.exception(e)
                raise e

    def run(self, ws: WSConnection):
        if self._status == "cancelled":
            return
        assert self._status == "stopped"

        task_queue = self._mp_ctx.Queue()
        result_queue = self._mp_ctx.Queue()

        if self._config.worker_role == "inference":
            child_target = inference_worker
        else:
            child_target = download_worker

        child_process = self._mp_ctx.Process(
            target=child_target,
            args=(
                task_queue,
                result_queue,
                self._task_runner_cls,
                self._config,
                self._sd_config,
                self._gpt_config,
            ),
        )
        child_process.start()

        self._status = "running"
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        try:
            task_producer_fut = pool.submit(self.task_producer, ws, task_queue)
            result_consumer_fut = pool.submit(self.result_consumer, ws, result_queue)
            done, _ = concurrent.futures.wait(
                [task_producer_fut, result_consumer_fut],
                return_when=concurrent.futures.FIRST_EXCEPTION,
            )
            has_error = False
            running_error: BaseException | None = None
            for fut in done:
                exc = fut.exception()
                if exc is not None:
                    has_error = True
                    running_error = exc

            if has_error:
                if child_process.is_alive():
                    child_process.kill()
                    _logger.info("close %s task forcely", self._config.worker_role)

                raise TaskWorkerRunningError from running_error
            else:
                if child_process.is_alive():
                    child_process.terminate()
                    _logger.info("close %s task gracefully", self._config.worker_role)
        except TaskWorkerRunningError:
            raise
        except Exception as e:
            _logger.error("Worker unexpected error")
            _logger.exception(e)
            if child_process.is_alive():
                child_process.kill()
                _logger.info("close %s task forcely", self._config.worker_role)
        finally:
            self._status = "stopped"
            pool.shutdown(wait=True, cancel_futures=True)
            child_process.join()
            _logger.info("%s task process is joined", self._config.worker_role)
            # Unflushed queue data must not block the parent process after
            # the child is gone
            for queue in (task_queue, result_queue):
                queue.close()
                queue.cancel_join_thread()
