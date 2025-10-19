import socketserver
import json
import os
import subprocess
import time
import traceback
import socket
import pathlib
import concurrent.futures
import tempfile
import shutil

# =====================
# Configuration
# =====================
HOST = "127.0.0.1"
PORT = 20543
HEADER_SIZE = 4
MAX_WORKERS = 8  # Maximum concurrent worker threads

# Configuration file path
CONFIG_FILE = (pathlib.Path(__file__).parent / "../../../data/log/serv_cfg.txt").resolve()
os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
with open(CONFIG_FILE, "w") as f:
    f.write(f"{HOST} {PORT}\n")

JOBS_BASE_DIR = (pathlib.Path(__file__).parent / "../../../data/local_jobs").resolve()

# =====================
# Task executor
# =====================
class TaskExecutor:
    """Single task executor that executes Python code in isolated temporary directories"""

    def __init__(self, task_id, code, timeout=120, ncore=1):
        self.task_id = task_id
        self.code = code
        self.timeout = timeout
        self.ncore = ncore
        self.result = None

    def execute(self):
        """Execute task and return results"""
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Write task code
                job_file = os.path.join(temp_dir, "job.py")
                with open(job_file, "w", encoding="utf-8") as f:
                    f.write(self.code)

                # Set environment variables
                env = os.environ.copy()
                env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__)) + ":" + env.get("PYTHONPATH", "")

                # CPU core limit: use taskset
                cmd = ["python", job_file]

                # Use taskset to limit CPU cores on Linux systems
                if os.name == 'posix' and shutil.which("taskset"):
                    # Select first ncore CPU cores
                    cpu_mask = (1 << self.ncore) - 1
                    cmd = ["taskset", f"0x{cpu_mask:x}"] + cmd

                # Execute task
                process = subprocess.run(
                    cmd,
                    cwd=temp_dir,
                    timeout=self.timeout,
                    capture_output=True,
                    text=True,
                    env=env
                )

                # Merge stdout and stderr, consistent with existing interface
                combined_output = process.stdout
                if process.stderr:
                    combined_output += "\n" + process.stderr

                self.result = {
                    "id": self.task_id,
                    "stdout": combined_output,
                    "success": process.returncode == 0
                }

        except subprocess.TimeoutExpired:
            self.result = {
                "id": self.task_id,
                "stdout": f"Task timed out after {self.timeout} seconds",
                "success": False
            }
        except Exception as e:
            self.result = {
                "id": self.task_id,
                "stdout": f"Execution error: {str(e)}",
                "success": False
            }

        return self.result

# =====================
# Job processing Handler
# =====================
class JobHandler(socketserver.BaseRequestHandler):
    def recv_with_header(self):
        """Read message with header"""
        header = self._recv_exact(HEADER_SIZE)
        if not header:
            return None
        msg_len = int.from_bytes(header, byteorder='big')
        return self._recv_exact(msg_len)

    def _recv_exact(self, n):
        """Read exactly specified number of bytes"""
        data = bytearray()
        while len(data) < n:
            packet = self.request.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return bytes(data)

    def handle(self):
        try:
            # Read request data
            data = self.recv_with_header()
            if not data:
                return

            # Parse request
            payload = json.loads(data.decode())
            batch_id = payload['batch_id']
            tasks = payload['tasks']
            timeout = payload['timeout']
            ncore = payload['ncore']

            # Create batch directory
            base_dir = f"{JOBS_BASE_DIR}/{time.strftime('%y%m%d_%H%M%S')}_{batch_id}"
            os.makedirs(base_dir, exist_ok=True)
            print(f"Received {len(tasks)} tasks for batch {batch_id}, timeout={timeout}s, ncore={ncore}")

            # Use thread pool to execute tasks
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit all tasks
                future_to_task = {}
                for task in tasks:
                    task_executor = TaskExecutor(
                        task_id=task['id'],
                        code=task['code'],
                        timeout=timeout,
                        ncore=ncore
                    )
                    future = executor.submit(task_executor.execute)
                    future_to_task[future] = task_executor

                # Wait for all tasks to complete
                results = []
                completed_count = 0

                for future in concurrent.futures.as_completed(future_to_task):
                    task_executor = future_to_task[future]
                    try:
                        result = future.result()
                        results.append(result)
                        completed_count += 1

                        # Save task results to file
                        task_dir = os.path.join(base_dir, result['id'])
                        os.makedirs(task_dir, exist_ok=True)

                        with open(os.path.join(task_dir, "stdout.txt"), "w") as f:
                            f.write(result['stdout'])

                        print(f"Task {result['id']} completed ({completed_count}/{len(tasks)})")

                    except Exception as e:
                        print(f"Task {task_executor.task_id} failed: {e}")
                        results.append({
                            "id": task_executor.task_id,
                            "stdout": f"Task execution failed: {str(e)}",
                            "success": False
                        })

            # Prepare response (consistent with existing interface: only id and stdout)
            response_results = []
            for result in results:
                response_results.append({
                    "id": result['id'],
                    "stdout": result['stdout']
                })

            response = {
                "batch_id": batch_id,
                "results": response_results
            }

            print(f"Batch {batch_id} completed with {len(results)} results")

            # Send response (consistent with other services, no header)
            self.request.setblocking(True)
            self.request.sendall(json.dumps(response).encode())

        except Exception as e:
            traceback.print_exc()
            error = {"error": f"{type(e).__name__}: {str(e)}"}
            try:
                self.request.setblocking(True)
                self.request.sendall(json.dumps(error).encode())
            except Exception:
                pass

# =====================
# Multi-threaded TCP server
# =====================
class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True

if __name__ == "__main__":
    # Ensure base directory exists
    os.makedirs(JOBS_BASE_DIR, exist_ok=True)

    server = ThreadedTCPServer((HOST, PORT), JobHandler)
    print(f"Local Server running on {HOST}:{PORT}")
    print(f"Max workers: {MAX_WORKERS}")
    print(f"Jobs directory: {JOBS_BASE_DIR}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer shutting down...")
        server.shutdown()
    except Exception as e:
        traceback.print_exc()