from typing import Optional

from meshcat.geometry import Geometry


class Cone(Geometry):
    def __init__(self, radius: float, height: float):
        super(Cone, self).__init__()
        self.radius = radius
        self.height = height

    def lower(self, object_data):
        return {
            "uuid": self.uuid,
            "type": "ConeGeometry",
            "radius": self.radius,
            "height": self.height,
        }


class PlaneGeometry(Geometry):
    def __init__(
        self,
        width: float,
        height: float,
        widthSegments: Optional[int] = None,
        heightSegments: Optional[int] = None,
    ):
        super(PlaneGeometry, self).__init__()
        self.width = width
        self.height = height
        self.widthSegments = widthSegments if widthSegments is not None else width
        self.heightSegments = heightSegments if heightSegments is not None else height

    def lower(self, object_data):
        return {
            "uuid": self.uuid,
            "type": "PlaneGeometry",
            "width": self.width,
            "height": self.height,
            "widthSegments": self.widthSegments,
            "heightSegments": self.heightSegments,
        }
