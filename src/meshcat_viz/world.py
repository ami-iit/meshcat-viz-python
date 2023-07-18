import pathlib
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
import rod
from scipy.spatial.transform import Rotation

from meshcat_viz.meshcat.light import HemisphereLight

from . import logging
from .fk.provider import FKProvider
from .meshcat.server import MeshCatServer
from .meshcat.visualizer import MeshcatVisualizer
from .model import MeshcatModel
from .model_builder import MeshcatModelBuilder


class MeshcatWorld:
    """High-level class to manage a MeshCat scene."""

    def __init__(self, draw_grid: bool = True) -> None:
        """Initialize a MeshCat world."""

        # The meshcat visualizer instance and the corresponding url
        self._web_url = None
        self._visualizer = None

        # A forward kinematics provider for each visualized model
        self._fk_provider: Dict[str, FKProvider] = dict()

        # All visualized models
        self._meshcat_models: Dict[str, MeshcatModel] = dict()

        # Options
        self.draw_grid = draw_grid

    def open(self) -> None:
        """Initialize the meshcat world by opening a visualizer."""

        _ = self.meshcat_visualizer

    def close(self) -> None:
        """Close the meshcat world and clear all the resources."""

        if self._visualizer is not None:
            # Close meshcat
            self.meshcat_visualizer.delete()
            self.meshcat_visualizer.close()

            # Clear local resources
            self._meshcat_models = dict()
            self._fk_provider = dict()
            self._visualizer = None

    def update_model(
        self,
        model_name: str,
        joint_positions: Optional[Sequence] = None,
        joint_names: Optional[List[str]] = None,
        base_position: Optional[Sequence] = None,
        base_quaternion: Optional[Sequence] = None,
    ) -> None:
        """
        Update a visualized model.

        Args:
            model_name: The name of the model to update.
            joint_positions: The joint positions to update.
            joint_names: The names of the joints to update.
            base_position: The base position to update.
            base_quaternion: The base quaternion to update.
        """

        if model_name not in self._meshcat_models:
            raise ValueError(model_name)

        # Store all the transforms to send to the meshcat visualizer.
        # We send them all together to minimize visualization artifacts.
        node_transforms: Dict[str, npt.NDArray] = dict()

        if base_position is not None:
            self._fk_provider[model_name].base_pose[0:3, 3] = base_position
            node_transforms[model_name] = self._fk_provider[model_name].base_pose

        if base_quaternion is not None:
            R = Rotation.from_quat(
                np.array(base_quaternion)[np.array([1, 2, 3, 0])]
            ).as_matrix()

            self._fk_provider[model_name].base_pose[0:3, 0:3] = R
            node_transforms[model_name] = self._fk_provider[model_name].base_pose

        if joint_positions is not None and np.array(joint_positions).size > 0:
            if len(joint_names) != len(joint_positions):
                raise ValueError(len(joint_names), len(joint_positions))

            for joint_name, position in zip(joint_names, joint_positions):
                if joint_name not in self._fk_provider[model_name].joint_positions:
                    raise ValueError(f"Unknown joint '{joint_name}'")

                self._fk_provider[model_name].joint_positions[joint_name] = position

            # Get the base transform
            B_H_W = np.linalg.inv(self._fk_provider[model_name].base_pose)

            # Store the base to link transform of all handled links
            for link_name in self._meshcat_models[model_name].link_to_node.keys():
                try:
                    W_H_L = self._fk_provider[model_name].get_frame_transform(
                        frame_name=link_name
                    )
                except Exception as e:
                    logging.warning(msg=str(e))
                    continue

                node_path = self._meshcat_models[model_name].get_node_path(
                    node_name=link_name
                )

                node_transforms[node_path] = B_H_W @ W_H_L

        # Send all the transforms in a single message
        self._meshcat_models[model_name].visualizer.set_transforms(
            paths=list(reversed(node_transforms.keys())),
            matrices=np.array(list(reversed(node_transforms.values())), dtype=float),
        )

    def insert_model(
        self,
        model_description: Union[str, pathlib.Path],
        is_urdf: Optional[bool] = None,
        model_name: Optional[str] = None,
        model_pose: Optional[Tuple[Sequence, Sequence]] = None,
        fk_provider: Optional[FKProvider] = None,
    ) -> str:
        """
        Insert a model in the world.

        Args:
            model_description: The model description either as a path to a URDF or SDF file
               or as a string with its content.
            is_urdf: Whether the model description is a URDF or SDF file.
            model_name: The name of the model. If not given, the name is extracted from
                the model description.
            model_pose: The initial pose of the model. If not given, the pose is extracted
                from the model description.
            fk_provider: The forward kinematics provider for the model. If not given,
                the forward kinematics is computed with iDynTree.

        Returns:
            The name of the inserted model.
        """

        # Create the ROD model from the SDF resource
        sdf = rod.Sdf.load(sdf=model_description, is_urdf=is_urdf)
        assert len(sdf.models()) == 1

        # Extract the first model
        rod_model = sdf.models()[0]

        # Check that a model name is available
        if model_name is None and rod_model.name in {None, ""}:
            raise ValueError("Failed to assign a name to the model")

        # Assign the model name
        model_name = model_name if model_name is not None else rod_model.name
        logging.debug(msg=f"Inserting model '{model_name}'")

        if model_name in self._meshcat_models.keys():
            raise ValueError(f"Model '{model_name}' is already part of the world")

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

        elif rod_model.pose is not None:
            meshcat_model.set_base_pose(transform=rod_model.pose.transform())

        # Initialize the FK provider
        if fk_provider is not None:
            self._fk_provider[model_name] = fk_provider

        else:
            model_description_string = (
                model_description
                if isinstance(model_description, str)
                else model_description.read_text()
            )

            if not is_urdf:
                from rod.urdf.exporter import UrdfExporter

                model_description_string = UrdfExporter.sdf_to_urdf_string(
                    sdf=sdf, pretty=True, gazebo_preserve_fixed_joints=False
                )

            from meshcat_viz.fk.idyntree_provider import IDynTreeFKProvider

            # Initialize the iDynTree provider
            self._fk_provider[model_name] = IDynTreeFKProvider(
                urdf=model_description_string,
                considered_joints=[
                    j.name
                    for j in rod.Sdf.load(sdf=model_description_string, is_urdf=True)
                    .models()[0]
                    .joints()
                    if j.type != "fixed"
                ],
            )

        # Store the model
        self._meshcat_models[meshcat_model.name] = meshcat_model

        return meshcat_model.name

    def remove_model(self, model_name: str) -> None:
        """
        Remove a model from the world.

        Args:
            model_name: The name of the model to remove.
        """

        if self._visualizer is None:
            msg = "The Meshcat visualizer hasn't been opened yet, the are no models"
            raise RuntimeError(msg)

        if model_name not in self._meshcat_models.keys():
            raise ValueError(f"Model '{model_name}' is not part of the visualization")

        self._meshcat_models[model_name].delete()
        self._meshcat_models.pop(model_name)

    @property
    def meshcat_visualizer(self) -> MeshcatVisualizer:
        """Build and return a lazy-initialized Meshcat visualizer."""

        if self._visualizer is not None:
            return self._visualizer

        # Start custom MeshCat server
        server_proc, zmq_url, self._web_url = MeshCatServer.start_as_subprocess()

        # Attach custom visualizer to custom server
        meshcat_visualizer = MeshcatVisualizer(zmq_url=zmq_url)
        meshcat_visualizer.window.server_proc = server_proc

        # Draw the terrain grid
        meshcat_visualizer["/Grid"].set_property("visible", self.draw_grid)

        # Configure the visualizer
        meshcat_visualizer["/Background"].set_property("visible", True)
        meshcat_visualizer["/Background"].set_property("top_color", [1, 1, 1])
        meshcat_visualizer["/Background"].set_property("bottom_color", [0, 0, 0])

        # Disable all the lights by default
        meshcat_visualizer["/Lights/SpotLight"].set_property("visible", False)
        meshcat_visualizer["/Lights/PointLightPositiveX"].set_property("visible", False)
        meshcat_visualizer["/Lights/PointLightNegativeX"].set_property("visible", False)
        meshcat_visualizer["/Lights/AmbientLight"].set_property("visible", False)
        meshcat_visualizer["/Lights/FillLight"].set_property("visible", False)

        # Enable only the PointLightNegativeX light
        meshcat_visualizer["/Lights/PointLightNegativeX"].set_property("visible", True)

        # Insert a Hemisphere light
        light = HemisphereLight(position=(0, 0, 5.0), intensity=1 / np.pi)
        meshcat_visualizer["/Lights/Hemisphere"].set_object(light)

        self._visualizer = meshcat_visualizer
        return self._visualizer

    @property
    def web_url(self) -> str:
        """Build a Meshcat visualizer and return its URL."""

        # Trigger the lazy creation of the visualizer
        _ = self.meshcat_visualizer

        # Return the corresponding URL
        return self._web_url
