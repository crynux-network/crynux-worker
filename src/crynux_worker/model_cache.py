from typing import Callable, Any

class ModelCache(object):
    def __init__(self):
        self.key: str | None = None
        self.value = None

    def load(self, key: str, model_loader: Callable[[], Any]):
        if self.key is None or self.key != key:
            del self.value

            self.key = key
            self.value = model_loader()
        return self.value