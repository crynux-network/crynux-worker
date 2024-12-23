import json
import logging
import os
import signal
from contextlib import contextmanager

import requests
import websockets.sync.client

from crynux_worker import version
from crynux_worker.config import (Config, generate_gpt_config,
                                  generate_sd_config, get_config)
from crynux_worker.task import HFTaskRunner, MockTaskRunner, TaskWorker

_logger = logging.getLogger(__name__)


def get_requests_proxy_url(proxy) -> str | None:
    if proxy is not None and proxy.host != "":

        if "://" in proxy.host:
            scheme, host = proxy.host.split("://", 2)
        else:
            scheme, host = "", proxy.host

        proxy_str = ""
        if scheme != "":
            proxy_str += f"{scheme}://"

        if proxy.username != "":
            proxy_str += f"{proxy.username}"

            if proxy.password != "":
                proxy_str += f":{proxy.password}"

            proxy_str += f"@"

        proxy_str += f"{host}:{proxy.port}"

        return proxy_str
    else:
        return None


@contextmanager
def requests_proxy_session(proxy):
    proxy_url = get_requests_proxy_url(proxy)
    if proxy_url is not None:
        origin_http_proxy = os.environ.get("HTTP_PROXY", None)
        origin_https_proxy = os.environ.get("HTTPS_PROXY", None)
        os.environ["HTTP_PROXY"] = proxy_url
        os.environ["HTTPS_PROXY"] = proxy_url
        try:
            yield {
                "http": proxy_url,
                "https": proxy_url,
            }
        finally:
            if origin_http_proxy is not None:
                os.environ["HTTP_PROXY"] = origin_http_proxy
            else:
                os.environ.pop("HTTP_PROXY")
            if origin_https_proxy is not None:
                os.environ["HTTPS_PROXY"] = origin_https_proxy
            else:
                os.environ.pop("HTTPS_PROXY")
    else:
        yield None


@contextmanager
def register_worker(version: str, worker_url: str, proxy=None):
    joined = False
    try:
        with requests_proxy_session(proxy=proxy) as proxies:
            resp = requests.post(f"{worker_url}/{version}", proxies=proxies)
        if resp.status_code != 200:
            err = resp.json()
            _logger.error(f"worker join error: {err}")
        else:
            joined = True
            _logger.info("worker join")
    except Exception as e:
        _logger.error("worker join unknown error")
        _logger.exception(e)

    try:
        yield
    finally:
        if joined:
            try:
                with requests_proxy_session(proxy=proxy) as proxies:
                    resp = requests.delete(f"{worker_url}/{version}", proxies=proxies)
                if resp.status_code != 200:
                    err = resp.json()
                    _logger.error(f"worker quit error: {err}")
                else:
                    _logger.info("worker quit")
            except Exception as e:
                _logger.error("worker quit unknown error")
                _logger.exception(e)


def worker(config: Config | None = None):
    if config is None:
        config = get_config()

    _version = version()

    _logger.info(f"Crynux worker version: {_version}")

    sd_config = generate_sd_config(config)
    gpt_config = generate_gpt_config(config)

    task_worker = TaskWorker(
        task_runner_cls=MockTaskRunner,
        config=config,
        sd_config=sd_config,
        gpt_config=gpt_config
    )
    _stop = False

    def _signal_handle(*args):
        _logger.info("terminate worker process")
        nonlocal _stop
        _stop = True
        task_worker.cancel()


    signal.signal(signal.SIGTERM, _signal_handle)

    while not _stop:
        with websockets.sync.client.connect(
            config.node_url
        ) as websocket, register_worker(_version, config.worker_url, config.proxy):
            version_msg = {"version": _version}
            websocket.send(json.dumps(version_msg))
            raw_init_msg = websocket.recv()
            assert isinstance(raw_init_msg, str)
            init_msg = json.loads(raw_init_msg)
            assert "worker_id" in init_msg
            worker_id = init_msg["worker_id"]

            _logger.info(f"Connected, worker id {worker_id}")

            task_worker.run(websocket)
