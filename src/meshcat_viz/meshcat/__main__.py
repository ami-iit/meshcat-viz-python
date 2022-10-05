from .server import MeshCatServer


def main():
    import argparse
    import asyncio
    import platform
    import sys
    import webbrowser

    # Fix asyncio configuration on Windows for Python 3.8 and above.
    # Workaround for https://github.com/tornadoweb/tornado/issues/2608
    if sys.version_info >= (3, 8) and platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    parser = argparse.ArgumentParser(
        description="Serve the MeshCat HTML files and listen for ZeroMQ commands"
    )
    parser.add_argument("--zmq-url", "-z", type=str, nargs="?", default=None)
    parser.add_argument("--open", "-o", action="store_true")
    parser.add_argument("--certfile", type=str, default=None)
    parser.add_argument("--keyfile", type=str, default=None)
    parser.add_argument(
        "--ngrok_http_tunnel",
        action="store_true",
        help="""
ngrok is a service for creating a public URL from your local machine, which
is very useful if you would like to make your meshcat server public.""",
    )
    results = parser.parse_args()
    bridge = MeshCatServer(
        zmq_url=results.zmq_url,
        certfile=results.certfile,
        keyfile=results.keyfile,
        ngrok_http_tunnel=results.ngrok_http_tunnel,
    )
    print("zmq_url={:s}".format(bridge.zmq_url))
    print("web_url={:s}".format(bridge.web_url))
    if results.open:
        webbrowser.open(bridge.web_url, new=2)

    try:
        bridge.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
