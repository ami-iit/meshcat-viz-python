from meshcat.geometry import ImageTexture, MeshPhongMaterial


class HeightmapMaterial(MeshPhongMaterial):
    def __init__(
        self,
        displacement_map: ImageTexture,
        displacement_scale: float,
        color=0xFFFFFF,
        reflectivity=0.5,
        map=None,
        side=2,
        transparent=None,
        opacity=1.0,
        linewidth=1.0,
        wireframe=False,
        wireframeLinewidth=1.0,
        vertexColors=False,
        **kwargs,
    ):
        self.displacementMap = displacement_map
        self.displacementScale = displacement_scale

        super().__init__(
            color=color,
            reflectivity=reflectivity,
            map=map,
            side=side,
            transparent=transparent,
            opacity=opacity,
            linewidth=linewidth,
            wireframe=wireframe,
            wireframeLinewidth=wireframeLinewidth,
            vertexColors=vertexColors,
            **kwargs,
        )

    # Copy-paste from `GenericMaterial.lower`, adding support of displacement* elements
    def lower(self, object_data):
        # Three.js allows a material to have an opacity which is != 1,
        # but to still be non-transparent, in which case the opacity only
        # serves to desaturate the material's color. That's a pretty odd
        # combination of things to want, so by default we juse use the
        # opacity value to decide whether to set transparent to True or
        # False.
        if self.transparent is None:
            transparent = bool(self.opacity != 1)
        else:
            transparent = self.transparent

        data = {
            "uuid": self.uuid,
            "type": self._type,
            "color": self.color,
            "reflectivity": self.reflectivity,
            "side": self.side,
            "transparent": transparent,
            "opacity": self.opacity,
            "linewidth": self.linewidth,
            "wireframe": bool(self.wireframe),
            "wireframeLinewidth": self.wireframeLinewidth,
            "vertexColors": (2 if self.vertexColors else 0),
            "displacementScale": self.displacementScale,
        }

        data.update(self.properties)

        if self.map is not None:
            data["map"] = self.map.lower_in_object(object_data)

        if self.displacementMap is not None:
            data["displacementMap"] = self.displacementMap.lower_in_object(object_data)

        return data
