from abc import ABC, abstractmethod
from typing import Any, Dict, Type

import numpy as np

from src.data.common_types import AbstractRawDataProvider
from src.utils import consts


def merge_two_dicts(x: Dict[str, Any], y: Dict[str, Any]) -> Dict[str, Any]:
    z = x.copy()
    z.update(y)
    return z


class EstimatorModel(ABC):
    @abstractmethod
    def get_model_fn(self):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def summary(self) -> str:
        pass

    @property
    def params(self) -> Dict[str, Any]:
        base_params = {consts.DATA_PROVIDER_CLS: self.dataset_provider_cls}
        return merge_two_dicts(base_params, self.additional_model_params)

    @property
    @abstractmethod
    def additional_model_params(self) -> Dict[str, Any]:
        return {}

    @property
    @abstractmethod
    def dataset_provider_cls(self) -> Type[AbstractRawDataProvider]:
        pass

    @abstractmethod
    def get_predicted_labels(self, result: np.ndarray):
        pass

    @abstractmethod
    def get_predicted_scores(self, result: np.ndarray):
        pass
