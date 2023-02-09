from typing import Sequence

import meshcat.visualizer
import numpy.typing as npt

from .commands import SetTransforms


class MeshcatVisualizer(meshcat.visualizer.Visualizer):
    def __init__(
        self,
        zmq_url: str = None,
        window: meshcat.visualizer.ViewerWindow = None,
        server_args: Sequence = (),
    ):
        super(MeshcatVisualizer, self).__init__(
            zmq_url=zmq_url, window=window, server_args=server_args
        )

    def set_transforms(
        self, matrices: Sequence[npt.NDArray], paths: Sequence[str]
    ) -> None:
        set_transforms_cmd = SetTransforms(
            paths=paths, matrices=matrices, visualizer_root_path=self.path
        )

        self.window.zmq_socket.send_multipart(msg_parts=set_transforms_cmd.multipart())
        rec: str = self.window.zmq_socket.recv().decode("utf-8")

        if not rec.startswith("ok"):
            raise RuntimeError(f"[set_transforms] {rec}")
