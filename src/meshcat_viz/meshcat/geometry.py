from meshcat.geometry import Geometry


class Cone(Geometry):

    def __init__(self, radius: float, height: float):
        super(Cone, self).__init__()
        self.radius = radius
        self.height = height

    def lower(self, object_data):
        return {
            u"uuid": self.uuid,
            u"type": u"ConeGeometry",
            u"radius": self.radius,
            u"height": self.height,
        }
