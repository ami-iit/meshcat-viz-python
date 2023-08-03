import dataclasses
import logging
import tempfile
from typing import Optional, Tuple

import meshcat
import numpy as np
import numpy.typing as npt
import png

from . import logging


@dataclasses.dataclass
class Heightmap:
    """"""

    # This 2D matrix describes the (X, Y) -> Z terrain mapping.
    # The matrix refers to the x-axis and y-axis ranges determined by the
    # x_bounds and y_bounds attributes, discretized accordingly to the dimensions of
    # the matrix.
    matrix: npt.NDArray = dataclasses.field(kw_only=True)

    x_bounds: Tuple[float, float] = dataclasses.field(default=(-0.5, 0.5))
    y_bounds: Tuple[float, float] = dataclasses.field(default=(-0.5, 0.5))

    z_offset: float = dataclasses.field(default=None)
    z_bounds: Optional[Tuple[float, float]] = dataclasses.field(default=None)

    def to_grid(self) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """"""

        # Convert matrix from [0, 1] to [z_min, z_max]
        Z = self.matrix * (self.z_bounds[1] - self.z_bounds[0]) + self.z_offset

        # Create the x and y axes with absolute coordinates in [m], discretized
        # accordingly to the matrix dimensions
        x = np.linspace(self.x_bounds[0], self.x_bounds[1], self.matrix.shape[0])
        y = np.linspace(self.y_bounds[0], self.y_bounds[1], self.matrix.shape[1])

        # Create the X and Y matrices of the grid.
        # We aim to obtain the equivalence, for a generic i/j indices:
        # z = f(x[i], y[j]) = f(X[i, j], Y[i, j]) = Z[i, j].
        Y, X = np.meshgrid(y, x)

        return X, Y, Z

    def __post_init__(self):
        """"""

        if self.matrix.ndim != 2:
            raise ValueError(f"The matrix must be 2D (found {self.matrix.ndim} dims)")

        # Infer the z offset from the input matrix
        if self.z_offset is None:
            self.z_offset = np.min(self.matrix)

        # Infer the z bounds from the input matrix
        if self.z_bounds is None:
            self.z_bounds = (np.min(self.matrix), np.max(self.matrix))

        # Get the range of the z axis
        z_range = self.z_bounds[1] - self.z_bounds[0]

        # In the terrain is completely flat, z_range is zero
        z_range = z_range if z_range > 0 else 1.0

        # Store the matrix normalized in [0, 1]
        self.matrix = np.array(
            (self.matrix - self.z_offset) / z_range,
            dtype=np.float32,
        )

        assert not np.isnan(self.matrix).any()

    def to_meshcat(self) -> meshcat.geometry.ImageTexture:
        """Convert the matrix to ImageTexture compatible with MeshCat."""

        # The stored matrix is already normalized in [0, 1]
        matrix_01 = np.array(self.matrix, dtype=np.float32)

        # Convert to uint16 with maximum range
        matrix_uint16 = np.fmax(0, matrix_01 * 2**16 - 1).astype(np.uint16)

        # Covert the XY axes to image height and width
        H, W = matrix_01.shape

        # Create a temporary file with the PNG image...
        with tempfile.NamedTemporaryFile(mode="w+b", suffix=".png") as f:
            png.Writer(H, W, bitdepth=16, greyscale=True).write(
                f,
                # Note that we have o flip the matrix over the y axis to match the
                # convention used in meshcat
                np.fliplr(matrix_uint16).T.tolist(),
            )

            f.seek(0)
            png_image = meshcat.geometry.PngImage.from_file(f.name)

        # ... read by ImageTexture
        return meshcat.geometry.ImageTexture(image=png_image)

    def view(self) -> None:
        """Render the matrix as 2D greyscale image."""

        import matplotlib.image
        import matplotlib.pyplot as plt

        X, Y, Z = self.to_grid()

        x = X[:, 0]
        y = Y[0, :]

        dx = (x[1] - x[0]) / 2.0
        dy = (y[1] - y[0]) / 2.0
        extent = [x[0] - dx, x[-1] + dx, y[0] - dy, y[-1] + dy]

        fig, ax = plt.subplots(nrows=1, ncols=1)

        img: matplotlib.image.AxesImage = ax.imshow(
            Z.T, extent=extent, origin="lower", cmap="gray"
        )

        plt.colorbar(img)
        ax.set_title("Heightmap")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        plt.show()
