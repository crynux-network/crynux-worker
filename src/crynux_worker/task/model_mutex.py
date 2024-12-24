from contextlib import contextmanager
from multiprocessing.managers import SyncManager


class ModelMutex(object):
    def __init__(self, manager: SyncManager) -> None:
        self._condition = manager.Condition()
        self._dict = manager.dict()

    def accquire(self, model_id: str):
        with self._condition:
            while model_id in self._dict:
                self._condition.wait()
            self._dict[model_id] = None

    def release(self, model_id: str):
        with self._condition:
            assert model_id in self._dict
            del self._dict[model_id]
            self._condition.notify()

    @contextmanager
    def lock(self, model_id: str):
        self.accquire(model_id)
        try:
            yield
        finally:
            self.release(model_id)
