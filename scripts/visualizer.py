from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "visualizer" / "frontend"


def start(command: list[str], cwd: Path) -> subprocess.Popen:
    options = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    return subprocess.Popen(command, cwd=cwd, **options)


def stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the visualizer backend and frontend.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Reload the backend when Python files change (Vite always reloads the frontend).",
    )
    args = parser.parse_args()

    pnpm = shutil.which("pnpm")
    if pnpm is None:
        parser.error("pnpm is required; install it before starting the visualizer")

    backend_command = [
        sys.executable,
        "-m",
        "uvicorn",
        "visualizer.backend.app.main:app",
    ]
    if args.reload:
        backend_command.append("--reload")

    processes: list[subprocess.Popen] = []
    exit_code = 0
    try:
        processes.append(start(backend_command, ROOT))
        processes.append(start([pnpm, "dev"], FRONTEND))
        while all(process.poll() is None for process in processes):
            time.sleep(0.2)
        exit_code = next(
            (process.returncode for process in processes if process.returncode is not None),
            0,
        )
    except KeyboardInterrupt:
        pass
    finally:
        for process in processes:
            stop(process)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
