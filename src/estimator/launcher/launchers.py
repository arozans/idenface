from abc import ABC, abstractmethod
from typing import List, Iterator
from typing import TYPE_CHECKING

from dataclasses import dataclass

if TYPE_CHECKING:
    from src.estimator.model.estimator_model import EstimatorModel
from src.utils import consts


class ModelsIterator:
    _models: Iterator['EstimatorModel']
    current: 'EstimatorModel'

    def __init__(self, models: List['EstimatorModel']):
        self._models = iter(models)
        self.current = models[0]

    def __iter__(self):
        return self

    def __next__(self):
        self.current = self._models.__next__()
        return self.current


@dataclass
class RunData:
    model: 'EstimatorModel'
    launcher_name: str
    runs_directory_name: str
    is_experiment: bool
    run_no: int
    models_count: int


class Launcher(ABC):

    def __init__(self, models: List['EstimatorModel']):
        self.models = models

    @property
    def runs_data(self) -> List[RunData]:
        return [self._create_run_data(idx + 1, model) for idx, model in enumerate(self.models)]

    def _create_run_data(self, idx: int, model: 'EstimatorModel'):
        return RunData(
            model=model,
            launcher_name=self.launcher_name,
            runs_directory_name=self.runs_directory_name,
            is_experiment=self.is_experiment,
            run_no=idx,
            models_count=len(self.models)
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
    def launcher_name(self):
        pass


class DefaultLauncher(Launcher):
    def __init__(self, models: List['EstimatorModel']):
        if len(models) != 1:
            raise ValueError("DefaultLauncher should only run one 'EstimatorModel' instance")
        super().__init__(models)

    @property
    def is_experiment(self) -> bool:
        return False

    @property
    def runs_directory_name(self) -> str:
        return "models"

    @property
    def launcher_name(self):
        return self.models[0].name


class ExperimentLauncher(Launcher, ABC):
    @property
    def runs_directory_name(self) -> str:
        return consts.EXPERIMENT_LAUNCHER_RUNS_DIR_NAME

    @property
    def is_experiment(self) -> bool:
        return True
