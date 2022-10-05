import atexit
import os
import subprocess
import sys
from typing import List, Sequence, Tuple

from meshcat.servers.tree import find_node
from meshcat.servers.zmqserver import ZMQWebSocketBridge, match_web_url, match_zmq_url


class MeshCatServer(ZMQWebSocketBridge):
    def __init__(
        self,
        zmq_url=None,
        host="127.0.0.1",
        port=None,
        certfile=None,
        keyfile=None,
        ngrok_http_tunnel=False,
    ):

        super().__init__(
            zmq_url=zmq_url,
            host=host,
            port=port,
            certfile=certfile,
            keyfile=keyfile,
            ngrok_http_tunnel=ngrok_http_tunnel,
        )

    def handle_zmq(self, frames: Sequence[bytes]) -> None:

        cmd = frames[0].decode("utf-8")

        if cmd == "set_transforms":

            try:
                self.process_cmd_set_transforms(frames=frames)
                self.zmq_socket.send(f"ok {cmd}".encode("utf-8"))
                return

            except Exception as e:
                self.zmq_socket.send(f"error '{cmd}': {str(e)}".encode("utf-8"))
                return

        # Call the upstream method for processing default commands
        super().handle_zmq(frames=frames)

    def process_cmd_set_transforms(self, frames: Sequence[bytes]) -> None:

        # We cannot use individual calls like: super().handle_zmq(frames=frames)
        # since each of them would send back the ack message to the client.

        # Get the number of transforms
        size = int(frames[1].decode("utf-8"))

        # Make sure that the number of frames containing the command is correct
        if len(frames[2:]) != 2 * size:
            msg = f"error: expected {2 * size} frames"
            raise ValueError(msg)

        # Get the serialization of multiple SetTransform fields.
        # See the "meshcat_viz.meshcat.commands.SetTransforms" class for more details.
        set_transform_frames = frames[2:]

        # Get the node path and binary representation of the full SetTransform object
        paths = set_transform_frames[0::2]
        set_transform_lowered_cmds = set_transform_frames[1::2]

        # Iterate through the fields
        for path, set_transform_lowered in zip(paths, set_transform_lowered_cmds):

            # Create a hierarchical list of the node's path
            path_list = [p for p in path.decode("utf-8").split("/") if len(p) > 0]

            # Update the transform
            find_node(self.tree, path_list).transform = set_transform_lowered

            # Forward to websockets (otherwise the visualization doesn't refresh)
            super().forward_to_websockets(
                frames=("set_transform", path, set_transform_lowered)
            )

    @staticmethod
    def start_as_subprocess(
        zmq_url: str = None, server_args: List[str] = ()
    ) -> Tuple[subprocess.Popen, str, str]:

        # This is almost a copy of:
        #     meshcat.servers.zmqserver.start_zmq_server_as_subprocess
        # We had to re-define it here since the upstream server is started in a
        # different process, and monkey-patching it during runtime with the aim of
        # adding a new ZMQ command is not possible.
        # Therefore, we copied here the logic to start the server and this class
        # inherits from upstream by just extending ZMQWebSocketBridge.handle_zmq().

        # Need -u for unbuffered output: https://stackoverflow.com/a/25572491
        args = [sys.executable, "-u", "-m", "meshcat_viz.meshcat"]
        if zmq_url is not None:
            args.append("--zmq-url")
            args.append(zmq_url)
        if server_args:
            args.append(*server_args)
        # Note: Pass PYTHONPATH to be robust to workflows like Google Colab,
        # where meshcat might have been added directly via sys.path.append.
        # Copy existing environmental variables as some of them might be needed
        # e.g. on Windows SYSTEMROOT and PATH
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        # Use start_new_session if it's available. Without it, in jupyter the server
        # goes down when we cancel execution of any cell in the notebook.
        server_proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
        line = ""
        while "zmq_url" not in line:
            line = server_proc.stdout.readline().strip().decode("utf-8")
            if server_proc.poll() is not None:
                outs, errs = server_proc.communicate()
                print(outs.decode("utf-8"))
                print(errs.decode("utf-8"))
                raise RuntimeError(
                    "the meshcat server process exited prematurely with exit code "
                    + str(server_proc.poll())
                )
        zmq_url = match_zmq_url(line)
        web_url = match_web_url(server_proc.stdout.readline().strip().decode("utf-8"))

        def cleanup(server_proc):
            server_proc.kill()
            server_proc.wait()

        atexit.register(cleanup, server_proc)
        return server_proc, zmq_url, web_url
