import socketserver
import json
import os
import subprocess
import time
import traceback
import socket
import pathlib
import re

# =====================
# Configuration
# =====================
TASKS_PER_ROUND = 20                      # Number of tasks per queue per round
WAIT_AFTER_SUBMIT = 5                    # Wait time after each submission (seconds)
MAX_RETRIES = 8 * 60 * 6                # Number of retry rounds after bsub failure (8h)
QUEUES = ['batch_a100', 'batch_h100', 'batch_h200']

# Automatically calculate accessible IP
def get_accessible_ip():
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if not ip.startswith("127."):
            return ip
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        finally:
            s.close()
        return ip
    except:
        return "127.0.0.1"

HOST = get_accessible_ip()
PORT = 20543
HEADER_SIZE = 4

# Configuration file path
CONFIG_FILE = (pathlib.Path(__file__).parent / "../../../data/log/serv_cfg.txt").resolve()
os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
with open(CONFIG_FILE, "w") as f:
    f.write(f"{HOST} {PORT}\n")

JOBS_BASE_DIR = (pathlib.Path(__file__).parent / "../../../data/running_jobs").resolve()

# =====================
# Utility: Sanitize job name, avoid illegal characters
# =====================
def sanitize_job_name(s: str) -> str:
    # Keep only alphanumeric, underscore, dot and dash, replace others with underscore
    return re.sub(r'[^A-Za-z0-9_.-]', '_', str(s))

# =====================
# bsub job template (each array has its own directory array_dir)
# =====================
BSUB_ARRAY_TEMPLATE = """##!/bin/bash
#BSUB -J {job_name}[1-{ntasks}]
#BSUB -o {array_dir}/%I/stdout.txt
#BSUB -e {array_dir}/%I/stderr.txt
#BSUB -q {queue}
#BSUB -W {timeout_h}:{timeout_m}
#BSUB -n {ncore}
#BSUB -M 1024

hostname
export JAVA_TOOL_OPTIONS="-Djava.io.tmpdir={array_dir}/$LSB_JOBINDEX/.java/tmpdir -Dosgi.configuration.area={array_dir}/$LSB_JOBINDEX/.java/config -Dosgi.instance.area={array_dir}/$LSB_JOBINDEX/.java/instance"
export PATH=$PATH:$HOME/comsol63/opt/bin/

module load conda
conda activate exp

python {array_dir}/$LSB_JOBINDEX/job.py 2>&1
"""

# =====================
# Submit bsub job array (write bsub script to array_dir)
# =====================
def submit_bsub_array(array_dir, job_name, ntasks, timeout, queue, ncore):
    timeout_h, rem = divmod(timeout, 3600)
    timeout_m, timeout_s = divmod(rem, 60)
    if timeout_s >= 30:  # Round to Minutes
        timeout_m += 1

    bsub_script = BSUB_ARRAY_TEMPLATE.format(
        array_dir=array_dir,
        job_name=job_name,
        ntasks=ntasks,
        timeout_h=f"{timeout_h:02d}",
        timeout_m=f"{timeout_m:02d}",
        queue=queue,
        ncore=ncore
    )

    script_path = os.path.join(array_dir, "job_array.bsub")
    with open(script_path, "w") as f:
        f.write(bsub_script)


    # Use stdin to submit bsub (equivalent to bsub < job_array.bsub), retry on failure
    for attempt in range(MAX_RETRIES):
        result = subprocess.run(["bsub"], input=bsub_script, capture_output=True, text=True)
        if result.returncode == 0:
            try:
                job_id = result.stdout.split("<")[1].split(">")[0]
                return job_id
            except Exception:
                print(f"[ERROR] Cannot parse bsub return: {result.stdout}")
                return None
        else:
            if attempt % 60 == 0:  # Print failure log every minute
                print(f"[WARN] bsub submission failed (retried {attempt} times): {result.stderr.strip()}")
            time.sleep(10)

    print("[ERROR] bsub submission failed continuously for 8 hours, giving up.")
    return None

# =====================
# Check if tasks are pending (supports job array: use bjobs -a to expand subtasks)
# Returns: set of queue names containing PEND subtasks; empty set if no PEND; False if query fails
# =====================
def check_jobs_pending(job_ids):
    if not job_ids:
        return False
    result = subprocess.run(["bjobs", "-a", "-o", "jobid stat queue", "-noheader"] + job_ids,
                            capture_output=True, text=True)
    if result.returncode != 0:
        return False
    pending_queues = set()
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        # bjobs -a output: jobid stat queue (e.g. 12345[1] PEND batch_a100)
        if len(parts) >= 3:
            _, stat, queue = parts[0], parts[1], parts[2]
            if stat == "PEND":
                pending_queues.add(queue)
    return pending_queues

# =====================
# Job processing Handler
# =====================
class JobHandler(socketserver.BaseRequestHandler):
    def __init__(self, request, client_address, server):
        self.job_ids = []  # Save all submitted master job ids (job arrays)
        self.arrays_info = []  # Save metadata for each array: {job_id, array_dir, ntasks, mapping}
        super().__init__(request, client_address, server)

    def recv_with_header(self):
        header = self._recv_exact(HEADER_SIZE)
        if not header:
            return None
        msg_len = int.from_bytes(header, byteorder='big')
        return self._recv_exact(msg_len)

    def _recv_exact(self, n):
        data = bytearray()
        while len(data) < n:
            packet = self.request.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return bytes(data)

    def cancel_jobs(self):
        if not self.job_ids:
            return
        print(f"Terminating jobs: {','.join(self.job_ids)}")
        result = subprocess.run(["bkill"] + self.job_ids, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to terminate jobs: {result.stderr}")

    def check_job_completion(self):
        # Check if PEND/RUN tasks still exist by expanding all subtasks with bjobs -a
        self.request.setblocking(False)
        while True:
            if not self.job_ids:
                return True
            result = subprocess.run(["bjobs", "-a", "-o", "stat", "-noheader"] + self.job_ids,
                                    capture_output=True, text=True)
            # If query fails (e.g. job has been cleaned up), treat as completed
            if result.returncode != 0 or "not found" in result.stdout or not result.stdout.strip():
                return True
            states = set(line.strip() for line in result.stdout.strip().split("\n") if line.strip())
            # If no PEND or RUN, all subtasks have finished
            if not (states & {"PEND", "RUN"}):
                return True
            # Also check if client connection is still active
            try:
                sent = self.request.send(b'\x00')
                if sent == 0:
                    raise BrokenPipeError("Send zero bytes")
                data = self.request.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
                if data == b'':
                    raise ConnectionResetError("Client closed connection")
            except BlockingIOError:
                pass
            except (BrokenPipeError, ConnectionResetError) as e:
                print(f"[!] Connection disconnected: {type(e).__name__}")
                return False
            except OSError as e:
                if e.errno in (9, 107):
                    print(f"[!] Socket error: {e}")
                    return False
                raise
            time.sleep(1)

    def handle(self):
        data = self.recv_with_header()
        if not data:
            return

        try:
            payload = json.loads(data.decode())
            batch_id = payload['batch_id']
            tasks = payload['tasks']
            timeout = payload['timeout']
            ncore = payload['ncore']

            base_dir = f"{JOBS_BASE_DIR}/{time.strftime('%y%m%d_%H%M%S')}_{batch_id}"
            os.makedirs(base_dir, exist_ok=True)
            print(f"Received {len(tasks)} tasks")

            available_queues = list(QUEUES)
            task_index = 0
            queue_to_jobs = {q: [] for q in QUEUES}

            while task_index < len(tasks):
                if not available_queues:
                    print("[WARN] All queues are pending, waiting to retry...")
                    time.sleep(WAIT_AFTER_SUBMIT)
                    # Check recovered queues
                    for q in QUEUES:
                        pending_jobs = check_jobs_pending(queue_to_jobs[q])
                        if not pending_jobs:
                            available_queues.append(q)
                    continue

                for queue in available_queues:
                    # Get chunk to submit this round (length not exceeding TASKS_PER_ROUND)
                    chunk = tasks[task_index: task_index + TASKS_PER_ROUND]
                    if not chunk:
                        break

                    # job array  task  id
                    first_task_id = chunk[0]['id']
                    raw_job_name = f"{batch_id}_{first_task_id}"
                    job_name = sanitize_job_name(raw_job_name)
                    array_dir = os.path.join(base_dir, job_name)
                    os.makedirs(array_dir, exist_ok=True)

                    # Create index directory for each subtask in array 1..ntasks and write job.py
                    mapping = {}  # index_str -> original task id
                    for idx, task in enumerate(chunk, start=1):
                        idx_str = str(idx)
                        task_dir = os.path.join(array_dir, idx_str)
                        os.makedirs(task_dir, exist_ok=True)
                        job_py_path = os.path.join(task_dir, "job.py")
                        with open(job_py_path, "w") as f:
                            f.write(task['code'])
                        mapping[idx_str] = task['id']

                    ntasks = len(chunk)
                    # Submit job array (script placed in array_dir)
                    job_id = submit_bsub_array(array_dir, job_name, ntasks, timeout, queue, ncore)
                    if job_id:
                        self.job_ids.append(job_id)
                        queue_to_jobs[queue].append(job_id)
                        # Save array metadata
                        self.arrays_info.append({
                            "job_id": job_id,
                            "array_dir": array_dir,
                            "ntasks": ntasks,
                            "mapping": mapping,
                            "queue": queue
                        })
                    print(f"[INFO] Submitted {ntasks} tasks as job array {job_id} to queue {queue} (array_dir={array_dir}).")

                    task_index += ntasks
                    if task_index >= len(tasks):
                        break

                # Wait after submission completion
                time.sleep(WAIT_AFTER_SUBMIT)

                # Check which queues still have pending job array, skip these queues in next round
                available_queues = []
                for q in QUEUES:
                    pending_jobs = check_jobs_pending(queue_to_jobs[q])
                    if pending_jobs:
                        print(f"[INFO] Queue {q} has pending tasks, skipping next round")
                    else:
                        available_queues.append(q)

            if not self.job_ids:
                raise Exception("Failed to submit all jobs")

            print(f"Batch {batch_id} Submitted {len(self.job_ids)} job arrays: {self.job_ids}")

            if not self.check_job_completion():
                print(f"Client disconnected, terminating jobs {self.job_ids}")
                self.cancel_jobs()
                return

            print(f"Batch {batch_id} finished; collecting outputs...")

            # Collect stdout from all subtasks in arrays
            results = []
            for arr in self.arrays_info:
                array_dir = arr["array_dir"]
                mapping = arr["mapping"]
                for idx_str, orig_id in mapping.items():
                    out_path = os.path.join(array_dir, idx_str, "stdout.txt")
                    if os.path.exists(out_path):
                        try:
                            with open(out_path, "r") as f:
                                stdout = f.read()
                        except Exception as e:
                            stdout = f"<error reading stdout: {e}>"
                    else:
                        stdout = ""  # No output file may mean task hasn't been generated or has been cleaned up
                    results.append({"id": orig_id, "stdout": stdout})

            response = {"batch_id": batch_id, "results": results}
            self.request.setblocking(True)
            self.request.sendall(json.dumps(response).encode())
            self.request.setblocking(False)

        except Exception as e:
            traceback.print_exc()
            error = {"error": f"{type(e)}: {e}"}
            try:
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
    server = ThreadedTCPServer((HOST, PORT), JobHandler)
    print(f"BSUB Server running on {HOST}:{PORT}")
    try:
        server.serve_forever()
    except Exception as e:
        traceback.print_exc(e)
