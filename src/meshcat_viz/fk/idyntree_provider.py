import dataclasses
import pathlib
from typing import Dict, List, Union

import idyntree.bindings as idt
import numpy as np
import numpy.typing as npt

from .provider import FKProvider


@dataclasses.dataclass
class IDynTreeFKProvider(FKProvider):
    urdf: dataclasses.InitVar[Union[str, pathlib.Path]]
    considered_joints: dataclasses.InitVar[List[str]]

    base_pose: npt.NDArray = dataclasses.field(default=np.eye(4), init=False)
    joint_positions: Dict[str, float] = dataclasses.field(
        default_factory=dict, init=False
    )

    kin_dyn_computations: idt.KinDynComputations = dataclasses.field(
        default=None, init=False
    )

    def __post_init__(
        self, urdf: Union[str, pathlib.Path], considered_joints: List[str]
    ) -> None:
        # Read the URDF description
        urdf_string = urdf.read_text() if isinstance(urdf, pathlib.Path) else urdf

        # Create the model loader
        model_loader = idt.ModelLoader()

        # Load the URDF description
        if not model_loader.loadReducedModelFromString(urdf_string, considered_joints):
            raise RuntimeError(f"Failed to load URDF description")

        # Create KinDynComputations and insert the model
        kin_dyn = idt.KinDynComputations()

        # Load the model
        if not kin_dyn.loadRobotModel(model_loader.model()):
            raise RuntimeError("Failed to load model into KinDynComputations")

        # Store the object
        self.kin_dyn_computations = kin_dyn

        # Initialize the joint positions
        self.joint_positions = {name: 0.0 for name in considered_joints}

    def frame_exists(self, frame_name: str) -> bool:
        return self.kin_dyn_computations.getFrameIndex(frame_name) >= 0

    def get_frame_transform(self, frame_name: str) -> npt.NDArray:
        if not self.frame_exists(frame_name=frame_name):
            raise ValueError(f"Failed to find frame '{frame_name}'")

        H = idt.Transform()
        H.fromHomogeneousTransform(idt.Matrix4x4(self.base_pose))
        s = idt.VectorDynSize(list(self.joint_positions.values()))

        self.kin_dyn_computations.setRobotState(H, s, idt.Twist(), s, idt.Vector3())

        return (
            self.kin_dyn_computations.getWorldTransform(frame_name)
            .asHomogeneousTransform()
            .toNumPy()
        )
