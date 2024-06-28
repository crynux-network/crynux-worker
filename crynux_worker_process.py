import logging
import multiprocessing as mp
import pathlib
import platform
import signal
import time
import os
import sys
from typing import List

import requests
import whatthepatch

_logger = logging.getLogger(__name__)

version = "0.0.1"


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
    from crynux_worker import log
    from crynux_worker.config import get_config
    from crynux_worker.worker import worker
    from tenacity import Retrying, before_sleep_log, stop_never, wait_fixed

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

def _get_patch_contents(patch_url: str, current_version: str, platform_name: str):
    try:
        patch_contents = []
        resp = requests.get(patch_url + "/patches", params={"version": current_version, "platform": platform_name})
        resp.raise_for_status()

        data = resp.json()
        versions = data["versions"]
        if len(versions) > 0:
            urls = [f"{patch_url}/patch/{platform_name}/{v}" for v in versions]
            for url in urls:
                sub_resp = requests.get(url)
                sub_resp.raise_for_status()

                patch_contents.append(sub_resp.text)

        return versions, patch_contents
    except requests.HTTPError as e:
        _logger.exception(e)
        _logger.error("get patch contents error")
        return [], []


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

    patch_url = os.environ.get("CRYNUX_WORKER_PATCH_URL", "http://127.0.0.1:8000")
    version_file = os.environ.get("CRYNUX_WORKER_VERSION_FILE", "version.txt")

    platform_name = _get_platform()

    version_path = pathlib.Path(version_file)
    if version_path.exists():
        version = version_path.read_text().strip()
    else:
        version_path.write_text(version)

    ctx = mp.get_context("spawn")
    worker_process = ctx.Process(target=_worker)
    worker_process.start()

    def _is_worker_process_alive():
        try:
            return worker_process.is_alive()
        except ValueError:
            return False

    try:
        while True:
            remote_versions, patch_contents = _get_patch_contents(patch_url, version, platform_name)
            if len(remote_versions) > 0:
                for remote_version, patch_content in zip(
                    remote_versions, patch_contents
                ):
                    with DelayedKeyboardInterrupt():
                        apply_patch(patch_content)
                        with version_path.open(mode="w", encoding="utf-8") as f:
                            f.write(str(remote_version))
                        version = remote_version

                worker_process.terminate()
                worker_process.join()
                worker_process.close()
                worker_process = ctx.Process(target=_worker)
                worker_process.start()

            time.sleep(5)
    except KeyboardInterrupt:
        if _is_worker_process_alive():
            worker_process.kill()
            worker_process.join()
            worker_process.close()
