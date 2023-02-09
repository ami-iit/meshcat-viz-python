import pathlib
from typing import Dict, List, Optional, Sequence, Tuple, Union

import jax
import jaxsim.high_level.model
import numpy as np
import rod

from .meshcat.server import MeshCatServer
from .meshcat.visualizer import MeshcatVisualizer
from .model import MeshcatModel
from .model_builder import MeshcatModelBuilder


class MeshcatWorld:
    def __init__(self, dt: float = 0.001, rtf: float = 1.0):
        self.dt = dt
        self.rtf = rtf

        self._visualizer = None

        self._fk = jax.jit(lambda m: m.forward_kinematics())
        self._meshcat_models: Dict[str, MeshcatModel] = dict()
        self._jaxsim_models: Dict[str, jaxsim.high_level.model.Model] = dict()

    def open(self) -> None:
        _ = self.meshcat_visualizer

    def close(self) -> None:
        if self._visualizer is not None:
            self.meshcat_visualizer.delete()

            self._jaxsim_models = dict()
            self._meshcat_models = dict()
            self._visualizer = None

    def update_model(
        self,
        model_name: str,
        joint_positions: Optional[Sequence] = None,
        joint_names: Optional[List[str]] = None,
        base_position: Optional[Sequence] = None,
        base_quaternion: Optional[Sequence] = None,
    ) -> None:
        if model_name not in self._meshcat_models:
            raise ValueError(model_name)

        if model_name not in self._jaxsim_models:
            raise ValueError(model_name)

        if base_position is not None:
            self._jaxsim_models[model_name].reset_base_position(
                position=np.array(base_position)
            )

        if base_quaternion is not None:
            self._jaxsim_models[model_name].reset_base_orientation(
                orientation=np.array(base_quaternion)
            )

        # Update transform of base link's node
        W_H_B = self._jaxsim_models[model_name].base_transform()
        self._meshcat_models[model_name].set_base_pose(transform=W_H_B)

        # TODO: use whole-body FK
        if joint_positions is not None:
            # Store the new joint configuration
            self._jaxsim_models[model_name].reset_joint_positions(
                positions=np.atleast_1d(joint_positions), joint_names=joint_names
            )

            # Compute forward kinematics with JaxSim
            W_H_i = self._fk(self._jaxsim_models[model_name])

            # In MeshCat, all link poses are relative to the model's base
            B_H_W = np.linalg.inv(W_H_B)
            B_H_i = B_H_W @ W_H_i

            # Update link transforms
            self._meshcat_models[model_name].set_link_transforms(
                link_names=self._jaxsim_models[model_name].link_names(),
                transforms=np.array(B_H_i, dtype=float),
            )

    def insert_model(
        self,
        model_description: Union[str, pathlib.Path],
        is_urdf: bool = False,
        model_name: str = None,
        model_pose: Optional[Tuple[Sequence, Sequence]] = None,
    ) -> str:
        # Create the ROD model from the SDF resource
        rod_model = rod.Sdf.load(sdf=model_description).model

        # Extract the model name if not given
        if model_name is None and rod_model.name not in {None, ""}:
            model_name = rod_model.name
        else:
            raise ValueError("Failed to assign a name to the model")

        if model_name in self._meshcat_models.keys():
            raise ValueError(f"Model '{model_name}' is already part of the world")

        # Create the JaxSim model from the SDF resource
        jaxsim_model = jaxsim.high_level.model.Model.build_from_sdf(
            sdf=model_description, model_name=model_name, is_urdf=is_urdf
        ).mutable(validate=True)

        # Create the MeshcatModel
        meshcat_model = MeshcatModelBuilder.from_rod_model(
            visualizer=self.meshcat_visualizer,
            rod_model=rod_model,
            model_name=model_name,
        )

        # Set the initial model pose
        if model_pose is not None:
            meshcat_model.set_base_pose(
                position=np.array(model_pose[0]), quaternion=np.array(model_pose[1])
            )

        # Store the model
        self._meshcat_models[meshcat_model.name] = meshcat_model

        # Store the JaxSim model, used to compute forward kinematics
        self._jaxsim_models[meshcat_model.name] = jaxsim_model

        return meshcat_model.name

    def remove_model(self, model_name: str) -> None:
        if self._visualizer is None:
            msg = "The Meshcat visualizer hasn't been opened yet, the are no models"
            raise RuntimeError(msg)

        if model_name not in self._meshcat_models.keys():
            raise ValueError(f"Model '{model_name}' is not part of the visualization")

        self._meshcat_models[model_name].delete()
        self._meshcat_models.pop(model_name)

    @property
    def meshcat_visualizer(self) -> MeshcatVisualizer:
        if self._visualizer is not None:
            return self._visualizer

        # Start custom MeshCat server
        server_proc, zmq_url, web_url = MeshCatServer.start_as_subprocess()

        # Attach custom visualizer to custom server
        meshcat_visualizer = MeshcatVisualizer(zmq_url=zmq_url)
        meshcat_visualizer.window.server_proc = server_proc

        # Configure the visualizer
        meshcat_visualizer["/Grid"].set_property("visible", True)
        meshcat_visualizer["/Background"].set_property("visible", True)
        meshcat_visualizer["/Background"].set_property("top_color", [1, 1, 1])
        meshcat_visualizer["/Background"].set_property("bottom_color", [0, 0, 0])

        self._visualizer = meshcat_visualizer
        return self._visualizer
