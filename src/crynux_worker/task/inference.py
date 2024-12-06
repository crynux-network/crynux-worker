import os
import json
import logging
import signal
import socket
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from multiprocessing import get_context
from multiprocessing.connection import Connection
from selectors import EVENT_READ, DefaultSelector
from threading import Thread
from typing import Type, Literal

from gpt_task.config import Config as GPTConfig
from pydantic import ValidationError
from sd_task.config import Config as SDConfig
from websockets.sync.connection import Connection as WSConnection

from crynux_worker.config import Config
from crynux_worker.model import (
    PayloadType,
    TaskInput,
    TaskType,
    WorkerPayloadMessage,
    WorkerPhase,
)
from crynux_worker.model_cache import ModelCache
from .runner import TaskRunner

_logger = logging.getLogger(__name__)


def _inference_one_task(
    task_runner: TaskRunner,
    task_input: TaskInput,
    model_cache: ModelCache,
    config: Config,
    sd_config: SDConfig,
    gpt_config: GPTConfig,
):

    try:
        output_dir = None
        if task_input.task_type == TaskType.SD_FT_LORA:
            output_dir = os.path.join(config.output_dir, task_input.task_id_commitment)
        results = task_runner.run_inference_task(
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


def _inference_process(
    pipe: Connection, task_runner_cls: Type[TaskRunner], config: Config, sd_config: SDConfig, gpt_config: GPTConfig
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

                def _signal_handle(*args):
                    nonlocal stop
                    stop = True
                    logging.info("try to inference process gracefully")

                signal.signal(signal.SIGTERM, _signal_handle)

                while not stop:
                    try:
                        if pipe.poll(0):
                            task_input: TaskInput = pipe.recv()
                            try:
                                data = _inference_one_task(
                                    task_runner=task_runner,
                                    task_input=task_input,
                                    model_cache=model_cache,
                                    config=config,
                                    sd_config=sd_config,
                                    gpt_config=gpt_config,
                                )
                                res = ("success", task_input, data)
                            except Exception as e:
                                _logger.exception(e)
                                tb = traceback.format_exc()
                                res = ("error", task_input, tb)

                            pipe.send(res)
                    except (EOFError, BrokenPipeError):
                        break
                logging.info("inference process exit normally")
    except KeyboardInterrupt:
        pass


InferenceTaskStatus = Literal["idle", "running", "cancelled", "finished"]


class InferenceTask(object):
    def __init__(
        self, task_runner_cls: Type[TaskRunner], config: Config, sd_config: SDConfig, gpt_config: GPTConfig
    ) -> None:
        self._config = config
        self._sd_config = sd_config
        self._gpt_config = gpt_config

        ctx = get_context("spawn")
        parent_pipe, child_pipe = ctx.Pipe()
        self._parent_pipe = parent_pipe
        self._child_pipe = child_pipe
        self._inference_process = ctx.Process(
            target=_inference_process, args=(child_pipe, task_runner_cls, config, sd_config, gpt_config)
        )

        self._monitor_thread = Thread(target=self._monitor)

        interrupt_read, interrupt_write = socket.socketpair()
        self._interrupt_read = interrupt_read
        self._interrupt_write = interrupt_write

        self._status: InferenceTaskStatus = "idle"

        self._selector = DefaultSelector()
        self._selector.register(self._interrupt_read, EVENT_READ)

    def _monitor(self):
        self._inference_process.join()
        _logger.info("inference process joined")
        self._inference_process.close()
        _logger.info("inference process closed")
        self._child_pipe.close()

    def start(self):
        if self._status == "cancelled":
            return
        assert self._status == "idle"
        self._inference_process.start()
        self._monitor_thread.start()
        self._status = "running"

    def cancel(self):
        if self._status != "cancelled" and self._status != "finished":
            _logger.info("cancel inference task")
            self._status = "cancelled"
            self._interrupt_write.send(b"\0")
            self._interrupt_write.close()

    def close(self):
        if self._monitor_thread.is_alive():
            self._monitor_thread.join()
        self._parent_pipe.close()
        self._selector.close()
        self._status = "finished"

    def _is_inference_process_alive(self):
        try:
            return self._inference_process.is_alive()
        except ValueError:
            return False

    def run_init_inference(self, websocket: WSConnection) -> bool:
        if self._status == "cancelled":
            return True

        assert self._status == "running", "Inference task status is not running"

        phase_msg = {"phase": "init_inference"}
        websocket.send(json.dumps(phase_msg))

        prompt = (
            "a realistic photo of an old man sitting on a brown chair, "
            "on the seaside, with blue sky and white clouds, a dog is lying "
            "under his legs, masterpiece, high resolution"
        )
        sd_inference_args = {
            "version": "2.0.0",
            "base_model": {"name": "crynux-ai/stable-diffusion-v1-5"},
            "prompt": prompt,
            "negative_prompt": "",
            "task_config": {
                "num_images": 1,
                "safety_checker": False,
                "cfg": 7,
                "seed": 99975892,
                "steps": 40,
            },
        }
        task_type = TaskType.SD
        task_input = TaskInput(
            task_id_commitment=bytes([0 for _ in range(32)]).hex(),
            task_name="inference",
            task_type=task_type,
            task_args=json.dumps(sd_inference_args),
        )

        _logger.info(
            f"Initial inference task starts at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        self._parent_pipe.send(task_input)

        while True:
            for key, _ in self._selector.select(0):
                if key.fileobj == self._interrupt_read:
                    self._inference_process.kill()
                    _logger.info("inference task is cancelled")
                    self._selector.unregister(self._interrupt_read)
                    self._interrupt_read.close()
                    return True
            if self._parent_pipe.poll(timeout=0):
                status, _, data = self._parent_pipe.recv()
                if status == "error":
                    _logger.info("Initial inference task error")
                    assert isinstance(data, str)
                    payload_msg = WorkerPayloadMessage(
                        worker_phase=WorkerPhase.InitInference,
                        has_payload=True,
                        has_next=False,
                        payload_type=PayloadType.Error,
                    )
                    websocket.send(payload_msg.model_dump_json())
                    websocket.send(data)
                    self._inference_process.kill()
                    return False
                else:
                    _logger.info(
                        f"Initial inference task completes at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    payload_msg = WorkerPayloadMessage(
                        worker_phase=WorkerPhase.InitInference,
                        has_payload=False,
                        has_next=False,
                    )
                    websocket.send(payload_msg.model_dump_json())
                    return True

    def run_inference(self, websocket: WSConnection):
        if self._status == "cancelled":
            return

        assert self._status == "running", "Inference task status is not running"

        phase_msg = {"phase": "inference"}
        websocket.send(json.dumps(phase_msg))

        try:
            while self._is_inference_process_alive():
                try:
                    raw_task_input = websocket.recv(0)
                    assert isinstance(raw_task_input, str)
                    if len(raw_task_input) > 0:
                        task_input = TaskInput.model_validate_json(raw_task_input)

                        _logger.info(
                            f"Inference task {task_input.task_id_commitment} starts at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        self._parent_pipe.send(task_input)
                    websocket.send("task received")
                except TimeoutError:
                    pass

                for key, _ in self._selector.select(0):
                    if key.fileobj == self._interrupt_read:
                        self._inference_process.terminate()
                        _logger.info("inference task is cancelled")
                        self._selector.unregister(self._interrupt_read)
                        self._interrupt_read.close()
                if self._parent_pipe.poll(0):
                    status, task_input, data = self._parent_pipe.recv()
                    if status == "success":
                        _logger.info(
                            f"Inference task {task_input.task_id} completes at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        if task_input.task_type == TaskType.SD:
                            payload_type = PayloadType.PNG
                        elif task_input.task_type == TaskType.LLM:
                            payload_type = PayloadType.Json
                        else:
                            payload_type = PayloadType.Text

                        for i, res in enumerate(data):
                            payload_msg = WorkerPayloadMessage(
                                worker_phase=WorkerPhase.Inference,
                                has_payload=True,
                                has_next=i < len(data) - 1,
                                payload_type=payload_type,
                            )
                            websocket.send(payload_msg.model_dump_json())
                            websocket.send(res)
                    else:
                        assert isinstance(data, str)
                        _logger.error(f"Inference task {task_input.task_id} error")
                        payload_msg = WorkerPayloadMessage(
                            worker_phase=WorkerPhase.Inference,
                            has_payload=True,
                            has_next=False,
                            payload_type=PayloadType.Error,
                        )
                        websocket.send(payload_msg.model_dump_json())
                        websocket.send(data)
        except Exception as e:
            _logger.exception(e)
            _logger.info("inference task error")
            self._inference_process.kill()
            return
