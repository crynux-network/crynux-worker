import json
import logging
import signal

import websockets.sync.client

from crynux_worker.config import Config, get_config, generate_sd_config, generate_gpt_config
from crynux_worker.task import PrefetchTask, InferenceTask

_logger = logging.getLogger(__name__)

version = "0.0.1"

def worker(
    config: Config | None = None
):
    if config is None:
        config = get_config()

    _logger.info(f"Crynux worker version: {version}")

    sd_config = generate_sd_config(config)
    gpt_config = generate_gpt_config(config)
    
    total_models = 0
    if sd_config.preloaded_models.base is not None:
        total_models += len(sd_config.preloaded_models.base)
    if gpt_config.preloaded_models.base is not None:
        total_models += len(gpt_config.preloaded_models.base)
    if sd_config.preloaded_models.controlnet is not None:
        total_models += len(sd_config.preloaded_models.controlnet)
    if sd_config.preloaded_models.vae is not None:
        total_models += len(sd_config.preloaded_models.vae)
    prefetch_task = PrefetchTask(sd_config=sd_config, gpt_config=gpt_config, total_models=total_models)

    inference_task = InferenceTask(sd_config=sd_config, gpt_config=gpt_config)

    def _signal_handle(*args):
        _logger.info("terminate worker process")
        prefetch_task.cancel()
        inference_task.cancel()

    signal.signal(signal.SIGTERM, _signal_handle)

    with websockets.sync.client.connect(config.node_url) as websocket:
        version_msg = {"version": version}
        websocket.send(json.dumps(version_msg))
        raw_init_msg = websocket.recv()
        assert isinstance(raw_init_msg, str)
        init_msg = json.loads(raw_init_msg)
        assert "worker_id" in init_msg
        worker_id = init_msg["worker_id"]

        _logger.info(f"Connected, worker id {worker_id}")

        # prefetch
        success = prefetch_task.run(websocket)
        if not success:
            raise ValueError("prefetching models failed")

        inference_task.start()
        try:
            # init inference
            success = inference_task.run_init_inference(websocket)
            if not success:
                raise ValueError("init inferece task failed")

            # inference
            inference_task.run_inference(websocket)
        finally:
            inference_task.close()