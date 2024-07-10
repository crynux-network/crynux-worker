import logging
import multiprocessing as mp
import os
import pathlib
import platform
import signal
import sys
import time
from contextlib import contextmanager
from multiprocessing.connection import Connection
from multiprocessing.context import SpawnContext
from packaging.version import Version
from typing import Dict

import requests
import whatthepatch

from crynux_worker.config import get_config

_logger = logging.getLogger(__name__)


def _version(pipe: Connection):
    from crynux_worker import version

    _version = version()
    pipe.send(_version)


def get_version(ctx: SpawnContext):
    parent_pipe, child_pipe = ctx.Pipe()
    p = ctx.Process(target=_version, args=(child_pipe,))
    p.start()
    p.join()
    v = parent_pipe.recv()
    child_pipe.close()
    parent_pipe.close()
    p.close()
    return v


def apply_patch(patch_content: str):
    diffs = list(whatthepatch.parse_patch(patch_content))
    for diff in diffs:
        assert diff.header is not None

        old_path = pathlib.Path(diff.header.old_path)
        if old_path.exists():
            old_content = old_path.read_text()
        else:
            old_content = ""

        new_contents = whatthepatch.apply_diff(diff, old_content)
        if len(new_contents) > 0:
            old_path.parent.mkdir(parents=True, exist_ok=True)
            with old_path.open(mode="w", encoding="utf-8") as f:
                for line in new_contents:
                    print(line, file=f)
        else:
            old_path.unlink()

            for dir_path in old_path.parents:
                if not any(dir_path.iterdir()):
                    dir_path.rmdir()
                else:
                    break


def _worker():
    from tenacity import Retrying, before_sleep_log, stop_never, wait_fixed

    from crynux_worker import log
    from crynux_worker.config import get_config
    from crynux_worker.worker import worker

    try:
        config = get_config()

        log.init(
            config.log.dir, log_level=config.log.level, log_filename=config.log.filename
        )

        for attemp in Retrying(
            stop=stop_never,
            wait=wait_fixed(10),
            before_sleep=before_sleep_log(
                logging.getLogger("crynux_worker"), logging.ERROR, True
            ),
        ):
            with attemp:
                worker(config)
    except KeyboardInterrupt:
        pass


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


def _get_patch_contents(
    patch_url: str, current_version: str, platform_name: str, proxy
) -> Dict[str, str]:
    current_major = Version(current_version).major
    with requests_proxy_session(proxy) as proxies:
        try:
            version_patches: Dict[str, str] = {}
            resp = requests.get(f"{patch_url}/patches.txt", proxies=proxies)
            resp.raise_for_status()

            versions = resp.text.strip().split()

            if len(versions) > 0:
                try:
                    index = versions.index(current_version)
                    versions = versions[index + 1 :]
                except ValueError:
                    pass

            for version in versions:
                major = Version(version).major
                if major == current_major:
                    sub_resp = requests.get(
                        f"{patch_url}/patches/{platform_name}/{version}.patch",
                        proxies=proxies,
                    )
                    sub_resp.raise_for_status()

                    version_patches[version] = sub_resp.text

            return version_patches
        except requests.RequestException as e:
            _logger.exception(e)
            _logger.error("get patch contents error")
            return {}


def _get_platform():
    if getattr(sys, "frozen", False):
        name = platform.system()
        if name in ["Darwin", "Windows", "Linux"]:
            return name.lower()
        else:
            raise ValueError(f"Unsupported platform: {name}")
    else:
        return "src"


class DelayedKeyboardInterrupt:

    def __enter__(self):
        self.signal_received = None
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received is not None and callable(self.old_handler):
            self.old_handler(*self.signal_received)


if __name__ == "__main__":
    mp.freeze_support()

    cfg = get_config()
    proxy = cfg.proxy

    patch_url = os.environ.get(
        "CRYNUX_WORKER_PATCH_URL",
        "https://raw.githubusercontent.com/crynux-ai/crynux-worker/main",
    )

    platform_name = _get_platform()

    ctx = mp.get_context("spawn")

    def _is_worker_process_alive():
        try:
            return worker_process.is_alive()
        except ValueError:
            return False

    try:
        version = get_version(ctx)
        version_patches = _get_patch_contents(
            patch_url, version, platform_name, proxy=proxy
        )
        if len(version_patches) > 0:
            for remote_version, patch_content in version_patches.items():
                with DelayedKeyboardInterrupt():
                    apply_patch(patch_content)
                _logger.info(f"update worker to version {remote_version}")

        worker_process = ctx.Process(target=_worker)
        worker_process.start()

        while True:
            version = get_version(ctx)
            version_patches = _get_patch_contents(
                patch_url, version, platform_name, proxy=proxy
            )
            if len(version_patches) > 0:
                for remote_version, patch_content in version_patches.items():
                    with DelayedKeyboardInterrupt():
                        apply_patch(patch_content)
                    _logger.info(f"update worker to version {remote_version}")

                worker_process.terminate()
                worker_process.join()
                worker_process.close()
                worker_process = ctx.Process(target=_worker)
                worker_process.start()

            time.sleep(60)
    except KeyboardInterrupt:
        if _is_worker_process_alive():
            worker_process.kill()
            worker_process.join()
            worker_process.close()
