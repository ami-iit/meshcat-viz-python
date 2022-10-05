import dataclasses
import itertools
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Sequence

import meshcat
import numpy as np
import numpy.typing as npt
from jaxsim import logging

from .meshcat.visualizer import MeshcatVisualizer


@dataclasses.dataclass
class MeshcatModel:

    name: str
    visualizer: MeshcatVisualizer = dataclasses.field(repr=False)

    NodeName = str
    LinkName = str

    # Dictionary from the link name defined in the SDF/URDF to the scoped named
    # used in the MeshCat visualizer
    link_to_node: Dict[LinkName, NodeName] = dataclasses.field(default_factory=dict)

    # Each link could have multiple visual shapes attached, each of them rigidly
    # attached to the link through a constant transforms.
    # We build the nodes tree in such a way that all visual shapes are automatically
    # moved when their corresponding link is moved.
    visual_shapes: DefaultDict[LinkName, List[NodeName]] = dataclasses.field(
        default_factory=defaultdict
    )

    def __post_init__(self):

        # Initialize the root node of the model
        self.visualizer[self.name].set_object(meshcat.geometry.Box([0.0, 0.0, 0.0]))

        if self.visual_shapes.default_factory not in {None, list}:
            logging.warning("Changing default factory of shapes dict to 'list'")

        if self.visual_shapes.default_factory is None:
            self.visual_shapes.default_factory = list

    def nodes_names(self, include_visual_nodes: bool = False) -> List[str]:

        model_nodes = [self.name] + list(self.link_to_node.values())

        if not include_visual_nodes:
            return model_nodes

        all_visual_shapes_nodes = list(
            itertools.chain.from_iterable(self.visual_shapes.values())
        )

        return model_nodes + all_visual_shapes_nodes

    def delete(self) -> None:

        self.visualizer[self.name].delete()

    def set_base_pose(
        self,
        transform: Optional[npt.NDArray] = None,
        position: Optional[npt.NDArray] = None,
        quaternion: Optional[npt.NDArray] = None,
    ) -> None:

        self.set_node_pose(
            node_name=self.name,
            transform=transform,
            position=position,
            quaternion=quaternion,
        )

    def set_link_pose(
        self,
        link_name: str,
        transform: Optional[npt.NDArray] = None,
        position: Optional[npt.NDArray] = None,
        quaternion: Optional[npt.NDArray] = None,
    ) -> None:

        if link_name not in self.link_to_node:
            raise ValueError(link_name)

        self.set_node_pose(
            node_name=self.link_to_node[link_name],
            transform=transform,
            position=position,
            quaternion=quaternion,
        )

    def set_node_pose(
        self,
        node_name: str,
        transform: Optional[npt.NDArray] = None,
        position: Optional[npt.NDArray] = None,
        quaternion: Optional[npt.NDArray] = None,
    ) -> None:

        if node_name not in self.nodes_names():
            raise ValueError(f"Failed to find node '{node_name}'")

        if {type(transform), type(position), type(quaternion)} == {None}:
            raise ValueError

        if transform is None and {type(position), type(quaternion)} == {None}:
            raise ValueError

        if position is None and quaternion is not None:
            position = np.zeros(4)

        if quaternion is None and position is not None:
            quaternion = np.array([1.0, 0, 0, 0])

        # Note: the transform of the node is always wrt its parent.
        # We build the graph so that the visual shapes are rigidly attached to their
        # parent link (therefore updating the link pose would move all its visuals),
        # and all links are children of the model's root (so that changing the root pose
        # moves the entire model).
        # This means that the link poses must be defined wrt the model's root.
        # The only exceptions are links considered as frames, in this case they are
        # rigidly attached to their parent link and their pose is constant.

        if transform is not None:
            parent_H_node = np.array(transform, dtype=float)

        elif position is not None and quaternion is not None:
            parent_H_node = meshcat.transformations.quaternion_matrix(
                quaternion=quaternion
            )
            parent_H_node[0:3, 3] = position

        else:
            raise RuntimeError

        # Update the node pose
        self.visualizer[node_name].set_transform(parent_H_node)

    def set_link_transforms(
        self, link_names: Sequence[str], transforms: Sequence[npt.NDArray]
    ) -> None:

        node_names = [self.link_to_node[l] for l in link_names]
        self.visualizer.set_transforms(paths=node_names, matrices=transforms)
