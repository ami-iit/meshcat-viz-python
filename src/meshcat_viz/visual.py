import dataclasses
import enum
import pathlib
from typing import Union

import meshcat
import numpy.typing as npt
import rod


class VisualShapeType(enum.IntEnum):

    Box = enum.auto()
    Capsule = enum.auto()
    Cylinder = enum.auto()
    Ellipsoid = enum.auto()
    Mesh = enum.auto()
    Plane = enum.auto()
    Sphere = enum.auto()


@dataclasses.dataclass
class VisualShapeData:

    name: str
    shape_type: VisualShapeType

    pose: npt.NDArray
    parent_link: rod.Link = dataclasses.field(repr=False)
    geometry: Union[rod.Box, rod.Sphere, rod.Cylinder, rod.Mesh]

    def scoped_name(self) -> str:

        return f"{self.parent_link.name}/{self.name}"

    def to_meshcat_geometry(self) -> meshcat.geometry.Geometry:

        if self.shape_type is VisualShapeType.Box:

            return meshcat.geometry.Box(self.geometry.size)

        elif self.shape_type is VisualShapeType.Sphere:

            return meshcat.geometry.Sphere(radius=self.geometry.radius)

        elif self.shape_type is VisualShapeType.Cylinder:

            return meshcat.geometry.Cylinder(
                height=self.geometry.length, radius=self.geometry.radius
            )

        elif self.shape_type is VisualShapeType.Mesh:

            assert pathlib.Path(self.geometry.uri).is_file()
            suffix = pathlib.Path(self.geometry.uri).suffix.lower()

            if suffix == ".dae":
                return meshcat.geometry.DaeMeshGeometry.from_file(self.geometry.uri)

            elif suffix == ".obj":
                return meshcat.geometry.ObjMeshGeometry.from_file(self.geometry.uri)

            elif suffix == ".stl":
                return meshcat.geometry.StlMeshGeometry.from_file(self.geometry.uri)

            raise ValueError(suffix)

        else:
            raise NotImplementedError(self.shape_type)
