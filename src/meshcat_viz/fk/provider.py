import abc
from typing import Dict

import numpy.typing as npt


class FKProvider(abc.ABC):
    base_pose: npt.NDArray
    joint_positions: Dict[str, float]

    @abc.abstractmethod
    def frame_exists(self, frame_name: str) -> bool:
        pass

    @abc.abstractmethod
    def get_frame_transform(self, frame_name: str) -> npt.NDArray:
        pass
