import dataclasses
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import jaxsim.typing
import numpy as np
import numpy.typing as npt
import rod
from jaxsim.high_level.model import Model

from .provider import FKProvider


@dataclasses.dataclass
class JaxSimFKProvider(FKProvider):
    rod_model: dataclasses.InitVar[rod.Model]
    considered_joints: dataclasses.InitVar[Optional[List[str]]] = dataclasses.field(
        default=None
    )

    base_pose: npt.NDArray = dataclasses.field(default=np.eye(4), init=False)
    joint_positions: Dict[str, float] = dataclasses.field(
        default_factory=dict, init=False
    )

    model: Model = dataclasses.field(default=None, init=False)

    def __post_init__(
        self, rod_model: rod.Model, considered_joints: Optional[List[str]]
    ) -> None:
        # Create a Sdf object
        sdf = rod.Sdf(model=rod_model)

        # Build a model
        self.model = Model.build_from_sdf(
            sdf=sdf.serialize(), considered_joints=considered_joints
        ).mutable(validate=True)

        # Extract the considered joints if not specified
        considered_joints = (
            considered_joints
            if considered_joints is not None
            else self.model.joint_names()
        )

        # Initialize the joint positions
        self.joint_positions = {name: 0.0 for name in considered_joints}

    def frame_exists(self, frame_name: str) -> bool:
        return frame_name in self.model.link_names()

    def get_frame_transform(self, frame_name: str) -> npt.NDArray:
        if not self.frame_exists(frame_name=frame_name):
            raise ValueError(f"Failed to find frame '{frame_name}'")

        self.model.reset_joint_positions(
            positions=jnp.array(list(self.joint_positions.values())),
            joint_names=list(self.joint_positions.keys()),
        )

        self.model.reset_base_transform(transform=jnp.array(self.base_pose))

        frame_idx = self.model.link_names().index(frame_name)
        return JaxSimFKProvider.fk(model=self.model)[frame_idx]

    @staticmethod
    @jax.jit
    def fk(model: jaxsim.high_level.model.Model) -> jaxsim.typing.ArrayJax:
        return model.forward_kinematics()
