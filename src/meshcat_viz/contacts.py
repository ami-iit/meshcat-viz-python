import dataclasses
from typing import Dict, Optional, Sequence

import meshcat
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from . import MeshcatModel
from .meshcat.geometry import Cone
from .meshcat.visualizer import MeshcatVisualizer


@dataclasses.dataclass
class ForceArrowNode:
    node_name: str
    visualizer: MeshcatVisualizer

    scale: float = dataclasses.field(default=1.0)
    radius: float = dataclasses.field(default=0.005)

    def __post_init__(self):
        # We create the following tree:
        #
        # contact_frame     (parent node)
        # └── arrow_origin  (node_name)
        #     ├── body
        #     └── tip
        #
        # We build an arrow 1m long in the z direction of the 'arrow_origin' node.
        #
        # Then, taking as input the 3D coordinates of the force expressed in
        # 'contact_frame', we update the arrow as follows:
        #
        # - direction: we compute the pure rotation between 'contact_frame' and the
        #              input force coordinates, and rotate 'arrow_origin' with it;
        # - magnitude: we apply a pure z scaling 'body' node and move the 'tip' node.

        # Create the body of the arrow
        self.visualizer[f"{self.node_name}/body"].set_object(
            geometry=meshcat.geometry.Cylinder(
                height=self.scale * 1.0,
                radius=self.radius,
            ),
            material=None,
        )

        # Compute the height and radius of the tip from the radius of the body
        cone_height = cone_radius = 1.5 * self.radius

        # Create the tip of the arrow
        self.visualizer[f"{self.node_name}/tip"].set_object(
            geometry=Cone(height=cone_height, radius=cone_radius),
            material=None,
        )

        # Initialize the arrow
        self.set(force=np.array([0, 0, 1.0]))

    def delete(self) -> None:
        self.visualizer[self.node_name].delete()

    def set(self, force: npt.NDArray) -> None:
        # Get the magnitude of the force
        magnitude = np.linalg.norm(force)

        # Hide the arrow if the force is zero
        hide = 0.0 if np.allclose(magnitude, 0.0) else 1.0

        # Objects in three.js are built with y-up instead of z-up
        R90x = Rotation.from_euler(seq="x", angles=np.pi / 2).as_matrix()

        # Compute the rotation that brings the z axis over the desired force direction
        R = (
            self.rotation_matrix_between(
                vector1=np.array([0, 0, 1]),
                vector2=force,
            )
            if not np.allclose(magnitude, 0.0)
            else np.eye(3)
        )

        # Build the transform
        H = np.eye(4)
        H[0:3, 0:3] = R

        # Rotate the 'arrow_origin' node.
        # Leave it as last transform in order to make hiding work.
        self.visualizer[self.node_name].set_transform(matrix=hide * H)

        # Scale the body along z and reposition the body
        self.visualizer[f"{self.node_name}/body"].set_transform(
            matrix=np.diag([1, 1, magnitude, 1])
            @ np.block([[R90x, np.vstack([0, 0, self.scale * 1.0 / 2])], [0, 0, 0, 1]])
        )

        cone_height = 1.5 * self.radius
        cone_position = np.vstack([0, 0, self.scale * magnitude + cone_height / 2])

        # Move the tip over the scaled body
        self.visualizer[f"{self.node_name}/tip"].set_transform(
            matrix=np.block([[R90x, cone_position], [0, 0, 0, 1]])
        )

    @staticmethod
    def rotation_matrix_between(
        vector1: npt.NDArray, vector2: npt.NDArray
    ) -> npt.NDArray:
        """
        Returns R such that: vector2 = R @ vector1.
        """

        # Normalize the vectors
        v1 = vector1 / np.linalg.norm(vector1)
        v2 = vector2 / np.linalg.norm(vector2)

        # Compute the axis of rotation between z and the force vector
        axis = np.cross(v1, v2)

        # Compute the sine and cosine of the angle between the vectors
        cosθ = np.dot(v1, v2)
        sinθ = np.linalg.norm(axis)

        # If the vectors are parallel, their cross product is zero.
        # Vectors could either point to the same direction or the opposite.
        if np.allclose(sinθ, 0.0):
            # Return the identity matrix if they are parallel with the same direction
            if cosθ > 0:
                return np.eye(3)

            # At this point, the vectors are parallel with opposite direction.
            # Select randomly a rotation axis.
            axis = np.array([1.0, 0, 0])

        # Build the skew-symmetric matrix associated to the axis
        x, y, z = axis
        K = np.array(
            [
                [0, -z, y],
                [z, 0, -x],
                [-y, x, 0],
            ]
        )

        # Return the rotation matrix computed with the Rodrigues formula
        return np.eye(3) + sinθ * K + (1 - cosθ) * K @ K


@dataclasses.dataclass
class ContactFrameNode:
    node_name: str

    visualizer: MeshcatVisualizer
    force: Optional[ForceArrowNode] = dataclasses.field(default=None, init=False)

    @property
    def name(self) -> str:
        return self.node_name.split("/")[-1]

    def __post_init__(self):
        # Add a new visualization dummy node
        self.visualizer[self.node_name].set_object(meshcat.geometry.Sphere(radius=0.0))

        # Set the initial transform
        self.set(transform=np.eye(4))

    def delete(self) -> None:
        self.visualizer[self.node_name].delete()

    def set(self, transform: npt.NDArray) -> None:
        # Set the transform between the reference node and the contact node
        self.visualizer[self.node_name].set_transform(transform)

    def contact_force(
        self,
        enabled: bool = True,
        scale: float = 1.0,
        radius: float = 0.005,
    ) -> None:
        if self.force:
            self.force.delete()
            self.force = None

        if not enabled:
            return

        self.force = ForceArrowNode(
            scale=scale,
            radius=radius,
            node_name=f"{self.node_name}/arrow_origin",
            visualizer=self.visualizer,
        )

        self.force.set(force=np.zeros(3))


@dataclasses.dataclass
class ModelContacts(Sequence):
    meshcat_model: MeshcatModel

    contact_frames: Dict[str, ContactFrameNode] = dataclasses.field(
        default_factory=dict, init=False
    )

    def __len__(self) -> int:
        return len(self.contact_frames)

    def __getitem__(self, item: str) -> ContactFrameNode:
        return self.contact_frames[item]

    def __contains__(self, item) -> bool:
        return item in self.contact_frames

    def __iter__(self):
        return iter(self.contact_frames.values())

    def __reversed__(self):
        return reversed(self.contact_frames.values())

    def add_contact_frame(
        self,
        contact_name: str,
        relative_to: str,
        transform: Optional[npt.NDArray] = None,
        enable_force: bool = True,
        force_scale: float = 1.0,
        force_radius: float = 0.005,
    ) -> None:
        if contact_name in self.contact_frames:
            raise ValueError(f"Contact node with name '{contact_name}' already exists")

        # Find reference node to attach the contact node.
        # It could either be the name of an existing link...
        if relative_to in self.meshcat_model.link_to_node:
            reference_node = self.meshcat_model.link_to_node[relative_to]
            node_name = f"{reference_node}/{contact_name}"

        # ... or the full path to an existing node.
        elif relative_to in self.meshcat_model.link_to_node.values():
            reference_node = relative_to
            node_name = f"{reference_node}/{contact_name}"

        else:
            raise ValueError(f"Failed to find force reference node '{relative_to}'")

        # Create the contact frame node
        contact_frame = ContactFrameNode(
            node_name=node_name,
            visualizer=self.meshcat_model.visualizer,
        )

        if enable_force:
            contact_frame.contact_force(
                enabled=True, scale=force_scale, radius=force_radius
            )

        # Store the contact frame node
        self.contact_frames[contact_name] = contact_frame

        # Apply the initial transform, if passed
        if transform is not None:
            self.contact_frames[contact_name].set(transform=transform)
