"""Small helpers so that notebooks can start and stop a FastAPI server in the background and wait until it answers.

start_server runs "python -m uvicorn <module>:app" as a separate process, writes its output to a log file and returns the process
handle once /health responds. stop_server ends it. This keeps the teaching notebooks self-contained: no second terminal needed.
"""
import os, sys, time, subprocess
import httpx


def start_server(module, port, env=None, log=None):
    log = log or f"server_{port}.log"
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", f"{module}:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        stdout=open(log, "w"), stderr=subprocess.STDOUT, env={**os.environ, **(env or {})})
    for _ in range(240):                                    # wait up to two minutes (model loading can take a while)
        if process.poll() is not None:
            raise RuntimeError("server exited:\n" + open(log).read()[-2000:])
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                return process
        except httpx.HTTPError:
            time.sleep(0.5)
    process.terminate()
    raise TimeoutError("server did not become healthy")


def stop_server(process):
    process.terminate()
    try:
        process.wait(10)
    except subprocess.TimeoutExpired:
        process.kill()
