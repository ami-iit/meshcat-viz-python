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
