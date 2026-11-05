import logging
import os
from logging.handlers import RotatingFileHandler
from typing import TextIO


class RotatingFileStream:
    """File-like stream with the same size rotation rules as RotatingFileHandler."""

    def __init__(self, filename: str, max_bytes: int, backup_count: int):
        if max_bytes < 1:
            raise ValueError("max_bytes must be >= 1")
        if backup_count < 1:
            raise ValueError("backup_count must be >= 1")
        self.base_filename = os.path.abspath(filename)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        dirname = os.path.dirname(self.base_filename)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        self._stream: TextIO = open(self.base_filename, mode="a", encoding="utf-8")

    def write(self, data: str) -> int:
        if not isinstance(data, str):
            data = str(data)
        written = self._stream.write(data)
        self._stream.flush()
        self._maybe_rollover()
        return written

    def flush(self) -> None:
        self._stream.flush()

    def close(self) -> None:
        if self._stream is not None and not self._stream.closed:
            self._stream.close()

    def fileno(self) -> int:
        return self._stream.fileno()

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def __enter__(self) -> "RotatingFileStream":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _maybe_rollover(self) -> None:
        self._stream.seek(0, os.SEEK_END)
        if self._stream.tell() < self.max_bytes:
            return
        self._do_rollover()

    def _do_rollover(self) -> None:
        self._stream.close()
        for i in range(self.backup_count - 1, 0, -1):
            src = f"{self.base_filename}.{i}"
            dst = f"{self.base_filename}.{i + 1}"
            if os.path.exists(src):
                if os.path.exists(dst):
                    os.remove(dst)
                os.rename(src, dst)
        backup = f"{self.base_filename}.1"
        if os.path.exists(backup):
            os.remove(backup)
        if os.path.exists(self.base_filename):
            os.rename(self.base_filename, backup)
        self._stream = open(self.base_filename, mode="a", encoding="utf-8")


def init(
    log_dir: str,
    log_level: str,
    max_bytes: int,
    backup_count: int,
    log_filename: str = "crynux-worker.log",
    root: bool = False,
):
    stream_handler = logging.StreamHandler()

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, log_filename)
    file_handler = RotatingFileHandler(
        log_file,
        encoding="utf-8",
        delay=True,
        maxBytes=max_bytes,
        backupCount=backup_count,
    )

    dt_fmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(
        "[{asctime}] [{levelname:<8}] {name}: {message}", dt_fmt, style="{"
    )

    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    if root:
        logger = logging.getLogger()
    else:
        logger = logging.getLogger("crynux_worker")

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(log_level)
