from pathlib import Path

from crynux_worker.log import RotatingFileStream


def test_rotating_file_stream_rolls_and_keeps_backup_count(tmp_path: Path):
    log_path = tmp_path / "crynux_worker_inference.log"
    max_bytes = 200
    backup_count = 2

    with RotatingFileStream(
        str(log_path), max_bytes=max_bytes, backup_count=backup_count
    ) as stream:
        chunk = "x" * 80
        for _ in range(20):
            stream.write(chunk)

    assert log_path.exists()
    assert log_path.stat().st_size < max_bytes + len(chunk)

    backup1 = Path(f"{log_path}.1")
    backup2 = Path(f"{log_path}.2")
    backup3 = Path(f"{log_path}.3")

    assert backup1.exists()
    assert backup2.exists()
    assert not backup3.exists()

    for path in (log_path, backup1, backup2):
        assert path.stat().st_size > 0


def test_rotating_file_stream_rejects_invalid_limits(tmp_path: Path):
    log_path = tmp_path / "bad.log"
    try:
        RotatingFileStream(str(log_path), max_bytes=0, backup_count=1)
        assert False, "expected ValueError for max_bytes < 1"
    except ValueError:
        pass
    try:
        RotatingFileStream(str(log_path), max_bytes=1, backup_count=0)
        assert False, "expected ValueError for backup_count < 1"
    except ValueError:
        pass
