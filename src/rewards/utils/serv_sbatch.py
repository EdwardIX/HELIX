import socketserver
import json
import os
import subprocess
import time
import shutil
import traceback
import socket

HOST, PORT = "0.0.0.0", 20543
HEADER_SIZE = 4

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={task_name}
#SBATCH --output={task_dir}/output.txt
#SBATCH --time={timeout_h:02d}:{timeout_m:02d}:{timeout_s:02d}
#SBATCH --cpus-per-task={ncore}

hostname
source ~/.bashrc
export PATH=$PATH:/workspace/tools/comsol63/bin/glnxa64/
export JAVA_TOOL_OPTIONS="-Djava.io.tmpdir={task_dir}/.java/tmpdir -Dosgi.configuration.area={task_dir}/.java/config -Dosgi.instance.area={task_dir}/.java/instance"

source "/workspace/tools/.conda/etc/profile.d/conda.sh"
conda activate comsol
# cd /workspace/data/comsol/geometry/v6.3geom # cd to geometry files path
cd /workspace/data/comsol/models/v6.3all
python {task_dir}/job.py
"""

def submit_sbatch(task_dir, task_name, timeout, ncore):
    timeout_h, timeout = divmod(timeout, 3600)
    timeout_m, timeout_s = divmod(timeout, 60)
    sbatch_script = SBATCH_TEMPLATE.format(
        task_dir = task_dir,
        task_name = task_name,
        timeout_h = timeout_h,
        timeout_m = timeout_m,
        timeout_s = timeout_s,
        ncore=ncore,
    )
    with open(f"{task_dir}/job.sbatch", "w") as f:
        f.write(sbatch_script)
    
    result = subprocess.run(["sbatch", f"{task_dir}/job.sbatch"], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return result.stdout.split()[-1]

# def check_job_completion(job_ids):
#     while True:
#         result = subprocess.run(["squeue", "-h", "--jobs", ",".join(job_ids)], 
#                               capture_output=True, text=True)
#         if not result.stdout.strip():
#             return True
#         time.sleep(1)

class JobHandler(socketserver.BaseRequestHandler):
    def __init__(self, request, client_address, server):
        self.job_ids = []  # ID
        # self.task_dirs = []  # 
        super().__init__(request, client_address, server)

    def recv_with_header(self):
        # 
        header = self._recv_exact(HEADER_SIZE)
        if not header:
            return None
        msg_len = int.from_bytes(header, byteorder='big')
        
        # 
        data = self._recv_exact(msg_len)
        return data
    
    def _recv_exact(self, n):
        """Read exactly specified number of bytes"""
        data = bytearray()
        while len(data) < n:
            packet = self.request.recv(n - len(data))
            if not packet:
                return None  # 
            data.extend(packet)
        return bytes(data)
    
    def cancel_jobs(self):
        """Cancel all submitted slurm jobs"""
        if not self.job_ids:
            return
            
        print(f"Terminating jobs: {','.join(self.job_ids)}")
        # Use scancel to terminate jobs
        result = subprocess.run(["scancel"] + self.job_ids,
                               capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to terminate jobs: {result.stderr}")

    def check_job_completion(self):
        """Improved connection state detection method"""
        # Set socket to non-blocking mode
        self.request.setblocking(False)
        
        while True:
            # Check if jobs are completed
            result = subprocess.run(
                ["squeue", "-h", "--jobs", ",".join(self.job_ids)],
                capture_output=True, text=True
            )
            if not result.stdout.strip():
                return True
            
            # 
            try:
                # 1: 
                sent = self.request.send(b'\x00')  # 
                if sent == 0:
                    raise BrokenPipeError("Send zero bytes")
                
                # 2: 
                # FIN
                data = self.request.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
                if data == b'':  # FIN
                    raise ConnectionResetError("Client closed connection")
                    
            except BlockingIOError:
                # 
                pass
            except (BrokenPipeError, ConnectionResetError) as e:
                print(f"[!] : {type(e).__name__}")
                return False
            except OSError as e:
                if e.errno in (9, 107):  # EBADF/ENOTCONN
                    print(f"[!] : {e}")
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
            
            base_dir = f"/workspace/.running_jobs/{time.strftime('%y%m%d_%H%M%S')}_{batch_id}"
            os.makedirs(base_dir, exist_ok=True)
            print(f"Received {len(tasks)} tasks")
            # job_ids = []
            task_map = {}
            for i in range(len(tasks)):
                task = tasks[i]
                task_id = task['id']
                task_dir = f"{base_dir}/{task_id}"
                os.makedirs(task_dir, exist_ok=True)
                
                with open(f"{task_dir}/job.py", "w") as f:
                    f.write(task['code'])
                
                job_id = submit_sbatch(task_dir, f"{batch_id}-{task_id}", timeout, ncore)
                if job_id:
                    self.job_ids.append(job_id)
                    task_map[job_id] = task_id
            
            if not self.job_ids:
                raise Exception("Failed to submit all jobs")
            print(f"Batch {batch_id} Submitted {len(self.job_ids)} jobs")
            
            if not self.check_job_completion():  # 
                print(f" {self.job_ids}")
                self.cancel_jobs()
                return
            
            print(f"Batch {batch_id} with {len(self.job_ids)} jobs finished")
            
            results = []
            for job_id in self.job_ids:
                task_id = task_map[job_id]
                output_file = f"{base_dir}/{task_id}/output.txt"
                if os.path.exists(output_file):
                    with open(output_file, "r") as f:
                        results.append({
                            "id": task_id,
                            "stdout": f.read()
                        })
            
            response = {"batch_id": batch_id, "results": results}
            self.request.setblocking(True)
            self.request.sendall(json.dumps(response).encode())
            self.request.setblocking(False)
        
        # except (BrokenPipeError, ConnectionResetError) as e:
        #     print(f": {self.job_ids}")
        #     self.cancel_jobs()  # 

        except Exception as e:
            error = {"error": f"{type(e)}: {e}"}
            self.request.sendall(json.dumps(error).encode())
    

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True

if __name__ == "__main__":
    server = ThreadedTCPServer((HOST, PORT), JobHandler)
    print(f"Server running on {HOST}:{PORT}")
    try:
        server.serve_forever()
    except Exception as e:
        traceback.print_exc(e)