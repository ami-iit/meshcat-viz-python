import dataclasses
from typing import Sequence

import meshcat.path
import numpy as np
import numpy.typing as npt
import umsgpack
from meshcat import commands


@dataclasses.dataclass
class SetTransforms:
    def __init__(
        self,
        matrices: Sequence[npt.NDArray],
        paths: Sequence[str],
        visualizer_root_path: meshcat.path.Path,
    ):

        self.paths = paths
        self.matrices = matrices
        self.visualizer_root_path = visualizer_root_path

        assert np.array(matrices).shape == (len(paths), 4, 4)

    def multipart(self) -> Sequence:

        tfs = [
            commands.SetTransform(matrix=m, path=self.visualizer_root_path.append(p))
            for (m, p) in zip(self.matrices, self.paths)
        ]

        cmd = b"set_transforms"
        size = str(len(self.paths)).encode("utf-8")

        multipart = [cmd, size]

        for tf in tfs:
            tf_lower = tf.lower()
            multipart += [
                tf_lower["path"].encode("utf-8"),
                umsgpack.packb(tf_lower),
            ]

        return multipart
