from abc import ABC, abstractmethod
from typing import List, Iterator, Dict, Any
from typing import TYPE_CHECKING

from dataclasses import dataclass

if TYPE_CHECKING:
    from src.estimator.model.estimator_conv_model import EstimatorConvModel
from src.utils import consts


class ModelsIterator:
    _models: Iterator['EstimatorConvModel']
    current: 'EstimatorConvModel'

    def __init__(self, models: List['EstimatorConvModel']):
        self._models = iter(models)
        self.current = models[0]

    def __iter__(self):
        return self

    def __next__(self):
        self.current = self._models.__next__()
        return self.current


@dataclass
class RunData:
    model: 'EstimatorConvModel'
    launcher_name: str
    runs_directory_name: str
    is_experiment: bool
    run_no: int
    models_count: int
    launcher_params: Dict[str, Any]

    def is_experiment_and_first_run(self):
        return self.is_experiment and self.run_no == 1


class Launcher(ABC):

    def __init__(self, models: List['EstimatorConvModel']):
        self.models = models

    @property
    def runs_data(self) -> List[RunData]:
        return [self._create_run_data(idx + 1, model) for idx, model in enumerate(self.models)]

    def _create_run_data(self, idx: int, model: 'EstimatorConvModel'):
        return RunData(
            model=model,
            launcher_name=self.name,
            runs_directory_name=self.runs_directory_name,
            is_experiment=self.is_experiment,
            run_no=idx,
            models_count=len(self.models),
            launcher_params=self.params
        )

    @property
    @abstractmethod
    def runs_directory_name(self) -> str:
        pass

    @property
    @abstractmethod
    def is_experiment(self) -> bool:
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    def params(self):
        return {}


class DefaultLauncher(Launcher):
    def __init__(self, models: List['EstimatorConvModel']):
        if len(models) != 1:
            raise ValueError("DefaultLauncher should only run one 'EstimatorConvModel' instance")
        super().__init__(models)

    @property
    def is_experiment(self) -> bool:
        return False

    @property
    def runs_directory_name(self) -> str:
        return consts.DEFAULT_LAUNCHER_RUNS_DIR_NAME

    @property
    def name(self):
        return self.models[0].name


class ExperimentLauncher(Launcher, ABC):
    @property
    def runs_directory_name(self) -> str:
        return consts.EXPERIMENT_LAUNCHER_RUNS_DIR_NAME

    @property
    def is_experiment(self) -> bool:
        return True
