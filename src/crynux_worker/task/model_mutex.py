from contextlib import contextmanager
from multiprocessing.managers import SyncManager
from typing import List


class ModelMutex(object):
    def __init__(self, manager: SyncManager) -> None:
        self._condition = manager.Condition()
        self._dict = manager.dict()

    def accquire_model(self, model_id: str):
        with self._condition:
            while model_id in self._dict:
                self._condition.wait()
            self._dict[model_id] = None

    def release_model(self, model_id: str):
        with self._condition:
            assert model_id in self._dict
            del self._dict[model_id]
            self._condition.notify_all()

    def accquire_models(self, model_ids: List[str]):
        with self._condition:
            while any(model_id in self._dict for model_id in model_ids):
                self._condition.wait()
            for model_id in model_ids:
                self._dict[model_id] = None

    def release_models(self, model_ids: List[str]):
        with self._condition:
            assert all(model_id in self._dict for model_id in model_ids)
            for model_id in model_ids:
                del self._dict[model_id]
            self._condition.notify_all()

    @contextmanager
    def lock_model(self, model_id: str):
        self.accquire_model(model_id)
        try:
            yield
        finally:
            self.release_model(model_id)

    @contextmanager
    def lock_models(self, model_ids: List[str]):
        self.accquire_models(model_ids)
        try:
            yield
        finally:
            self.release_models(model_ids)
