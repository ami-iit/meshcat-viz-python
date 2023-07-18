from typing import Tuple

import numpy as np
from meshcat.geometry import Object


class Light(Object):
    """Generic light object."""

    _type: str = None

    def __init__(self, **kwargs):
        super(Light, self).__init__(geometry=None, material=None)
        self.properties = kwargs

    def lower(self):
        data = {
            "metadata": {
                "version": 4.5,
                "type": "Object",
            },
            "object": {
                "uuid": self.uuid,
                "type": self._type,
            },
        }
        data["object"].update(self.properties)
        return data


class HemisphereLight(Light):
    _type = "HemisphereLight"

    def __init__(
        self,
        sky_color: int = 0xFFFFFF,
        ground_color: int = 0x444444,
        intensity: float = 1.0,
        position: Tuple[float, float, float] = (0, 0, 10.0),
    ) -> None:
        """"""

        self.position = position

        super(HemisphereLight, self).__init__(
            skyColor=sky_color,
            groundColor=ground_color,
            intensity=np.pi * intensity,
            position=list(position),
            castShadow=False,
        )


class DirectionalLight(Light):
    _type = "DirectionalLight"

    def __init__(
        self,
        color: int = 0xFFFFFF,
        intensity: float = 1.0,
        cast_shadow: bool = True,
        position: Tuple[float, float, float] = (0.0, 0.0, 10.0),
    ) -> None:
        self.position = position

        super(DirectionalLight, self).__init__(
            color=color,
            intensity=np.pi * intensity,
            castShadow=cast_shadow,
            position=list(position),
        )


class PointLight(Light):
    _type = "PointLight"

    def __init__(
        self,
        color: int = 0xFFFFFF,
        intensity: float = 1.0,
        distance: float = 0.0,
        decay: float = 2.0,
        cast_shadow: bool = True,
        position: Tuple[float, float, float] = (0.0, 0.0, 10.0),
    ) -> None:
        super(PointLight, self).__init__(
            color=color,
            intensity=np.pi * intensity,
            distance=distance,
            decay=decay,
            castShadow=cast_shadow,
            position=list(position),
        )
