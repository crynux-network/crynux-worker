import torch
import gc
import platform
from typing import Callable, Any


def get_accelerator():
    if platform.system() == "Darwin":
        try:
            import torch.mps

            return "mps"
        except ImportError:
            pass

    try:
        import torch.cuda

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    return "cpu"


class ModelCache(object):
    def __init__(self):
        self.key: str | None = None
        self.value = None

        self._accelerator = get_accelerator()

    def load(self, key: str, model_loader: Callable[[], Any]):
        if self.key is None or self.key != key:
            self.value = None
            self.key = None
            gc.collect()
            if self._accelerator == "cuda":
                torch.cuda.empty_cache()
            elif self._accelerator == "mps":
                torch.mps.empty_cache()

            self.value = model_loader()
            self.key = key
        return self.value