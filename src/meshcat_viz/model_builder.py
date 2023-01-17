import abc
import copy
import os
import pathlib
from typing import List

import meshcat
import numpy as np
import rod
import rod.kinematics
from jaxsim import logging
from scipy.spatial.transform import Rotation as R

from .meshcat.visualizer import MeshcatVisualizer
from .model import MeshcatModel
from .visual import VisualShapeData, VisualShapeType


class MeshcatModelBuilder(abc.ABC):
    @staticmethod
    def from_rod_model(
        visualizer: MeshcatVisualizer, rod_model: rod.Model, model_name: str = None
    ) -> MeshcatModel:

        # Resolve local URIs if present
        rod_model = MeshcatModelBuilder.resolve_sdf_tree_uris(rod_model=rod_model)

        # Build the MeshcatModel
        meshcat_model = MeshcatModel(name=model_name, visualizer=visualizer)

        # Extract the visual shapes from the SDF model
        visual_shapes = MeshcatModelBuilder.extract_visual_shapes(rod_model=rod_model)

        # Copy the links and frames kinematic tree.
        # Link transforms are relative to model's root.
        # Frame transforms are relative to the parent link of the frame.
        MeshcatModelBuilder.copy_tree(meshcat_model=meshcat_model, rod_model=rod_model)

        # Add the visual shapes. All their poses are relative to their parent link and,
        # similarly to frames, are constant and cannot be changed.
        _ = [
            MeshcatModelBuilder.add_visual_shape(
                meshcat_model=meshcat_model, visual_shape_data=vs
            )
            for vs in visual_shapes
        ]

        return meshcat_model

    @staticmethod
    def resolve_sdf_tree_uris(rod_model: rod.Model) -> rod.Model:

        rod_model_resolved = copy.deepcopy(rod_model)

        links: List[rod.Link] = (
            rod_model_resolved.link
            if isinstance(rod_model_resolved.link, list)
            else [rod_model_resolved.link]
        )

        for link in links:

            if link.visual is None:
                continue

            visuals: List[rod.Visual] = (
                link.visual if isinstance(link.visual, list) else [link.visual]
            )

            for visual in visuals:
                if visual.geometry.mesh is not None:

                    visual.geometry.mesh.uri = str(
                        MeshcatModelBuilder.resolve_local_uri(
                            uri=visual.geometry.mesh.uri
                        )
                    )

        return rod_model_resolved

    @staticmethod
    def resolve_local_uri(uri: str) -> pathlib.Path:

        # Remove the prefix of the URI
        uri_no_prefix = uri.split(sep="//")[-1]

        for path in os.environ["IGN_GAZEBO_RESOURCE_PATH"].split(":"):

            tentative = pathlib.Path(path) / uri_no_prefix

            if tentative.is_file():
                logging.debug(f"Resolved URI: {tentative}")
                return tentative

        raise RuntimeError(f"Failed to resolve URI: {uri}")

    @staticmethod
    def copy_tree(meshcat_model: MeshcatModel, rod_model: rod.Model) -> None:

        transforms = rod.kinematics.tree_transforms.TreeTransforms.build(
            model=rod_model, is_top_level=True
        )

        for link in transforms.kinematic_tree:

            if link.name() in meshcat_model.link_to_node:
                msg = f"Model '{meshcat_model.name}' already has link '{link.name()}'"
                raise ValueError(msg)

            # Build the name of the visualizer node
            node_name = f"{meshcat_model.name}/{link.name()}"
            logging.debug(f"Adding link visualization node '{node_name}'")

            # Add a new visualization node under the root node
            meshcat_model.visualizer[node_name].set_object(
                meshcat.geometry.Box([0.0, 0.0, 0.0])
            )

            # Store the link name
            meshcat_model.link_to_node[link.name()] = node_name

            # Compute the node transform wrt the tree root node (model base)
            root_H_link = transforms.relative_transform(
                relative_to=transforms.kinematic_tree.link_names()[0], name=link.name()
            )

            # Store the node transforms
            meshcat_model.visualizer[node_name].set_transform(root_H_link)

        for frame in transforms.kinematic_tree.frames:

            if frame.name() in meshcat_model.frame_to_node:
                msg = f"Model '{meshcat_model.name}' already has frame '{frame.name()}'"
                raise ValueError(msg)

            all_attachables_nodes_dict = dict(
                **meshcat_model.frame_to_node,
                **meshcat_model.link_to_node,
            )

            if frame.attached_to() not in all_attachables_nodes_dict:
                msg = "Failed to find parent element '{}' of frame '{}'"
                logging.warning(msg=msg.format(frame.attached_to(), frame.name()))
                continue

            # Build the name of the visualizer node
            node_name = (
                f"{all_attachables_nodes_dict[frame.attached_to()]}/{frame.name()}"
            )
            logging.debug(f"Adding frame visualization node '{node_name}'")

            # Add a new visualization node under the parent's link node
            meshcat_model.visualizer[node_name].set_object(
                meshcat.geometry.Box([0.0, 0.0, 0.0])
            )

            # Store the frame name
            meshcat_model.frame_to_node[frame.name()] = node_name

            # Initialize the fixed node transform
            parent_H_frame = transforms.relative_transform(
                relative_to=frame.attached_to(), name=frame.name()
            )
            meshcat_model.visualizer[node_name].set_transform(parent_H_frame)

    @staticmethod
    def extract_visual_shapes(rod_model: rod.Model) -> List[VisualShapeData]:

        visual_shapes: List[VisualShapeData] = list()

        links: List[rod.Link] = (
            rod_model.link if isinstance(rod_model.link, list) else [rod_model.link]
        )

        for link in links:

            if link.visual is None:
                continue

            visuals: List[rod.Visual] = (
                link.visual if isinstance(link.visual, list) else [link.visual]
            )

            for visual in visuals:

                # Get the transform from the parent link to the visual shape
                if visual.pose is None:
                    link_H_shape = np.eye(4)
                else:
                    link_H_shape = visual.pose.transform()
                    assert visual.pose.relative_to in {None, "", link.name}

                # Initialize the visual shape data
                visual_shape_data = None

                if visual.geometry.box is not None:

                    visual_shape_data = VisualShapeData(
                        name=visual.name,
                        shape_type=VisualShapeType.Box,
                        geometry=visual.geometry.box,
                        pose=link_H_shape,
                        parent_link=link,
                    )

                elif visual.geometry.cylinder is not None:

                    # three.js aligns the y-axis with the axis of rotational symmetry,
                    # instead SDF aligns with the z axis
                    static_tf = np.eye(4)
                    static_tf[0:3, 0:3] = R.from_euler(
                        seq="x", degrees=True, angles=90.0
                    ).as_matrix()

                    visual_shape_data = VisualShapeData(
                        name=visual.name,
                        shape_type=VisualShapeType.Cylinder,
                        geometry=visual.geometry.cylinder,
                        pose=link_H_shape @ static_tf,
                        parent_link=link,
                    )

                    visual_shapes.append(visual_shape_data)

                elif visual.geometry.sphere is not None:

                    visual_shape_data = VisualShapeData(
                        name=visual.name,
                        shape_type=VisualShapeType.Sphere,
                        geometry=visual.geometry.sphere,
                        pose=link_H_shape,
                        parent_link=link,
                    )

                elif visual.geometry.mesh is not None:

                    # We need to apply the scale, if defined
                    if visual.geometry.mesh.scale is not None:
                        scale = np.array(visual.geometry.mesh.scale)
                        link_H_shape = link_H_shape @ np.diag(np.hstack([scale, 1.0]))

                    visual_shape_data = VisualShapeData(
                        name=visual.name,
                        shape_type=VisualShapeType.Mesh,
                        geometry=visual.geometry.mesh,
                        pose=link_H_shape,
                        parent_link=link,
                    )

                # Add the shape to the list
                if visual_shape_data is not None:
                    visual_shapes.append(visual_shape_data)

        return visual_shapes

    @staticmethod
    def add_visual_shape(
        meshcat_model: MeshcatModel, visual_shape_data: VisualShapeData
    ) -> None:

        # Get the parent link of the visual shape
        parent_link_name = visual_shape_data.parent_link.name

        # if visual_shape_data.scoped_name() in meshcat_model.visual_shapes:
        if visual_shape_data.name in meshcat_model.link_to_node[parent_link_name]:
            msg = "Link '{}' already has visual '{}'"
            raise ValueError(msg.format(parent_link_name, visual_shape_data.name))

        # Check that the parent link is already a visualized node
        if parent_link_name not in meshcat_model.link_to_node:
            msg = "Failed to find link '{}', parent of shape '{}'"
            raise ValueError(msg.format(parent_link_name, visual_shape_data.name))

        # Build the name of the visualizer node
        node_name = (
            f"{meshcat_model.link_to_node[parent_link_name]}/{visual_shape_data.name}"
        )

        # Add the visual shape node under the link node
        meshcat_model.visualizer[node_name].set_object(
            visual_shape_data.to_meshcat_geometry()
        )

        # Store the visual shape name as dict entry
        meshcat_model.visual_shapes[parent_link_name] += [node_name]

        # Initialize the node transform
        meshcat_model.visualizer[node_name].set_transform(visual_shape_data.pose)
