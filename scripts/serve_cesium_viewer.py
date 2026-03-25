from __future__ import annotations

from pathlib import Path
import http.server
import socketserver


def main() -> None:
    viewer_dir = Path("outputs/viewer").resolve()
    port = 8765
    handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(*args, directory=str(viewer_dir), **kwargs)
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        print(f"Serving Cesium viewer assets from {viewer_dir}")
        print(f"Open http://127.0.0.1:{port}/cesium_global_magnetic_globe.html")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
