import socket
import json
import time
import os
import pathlib

SERVER_CONFIG_PATH = (pathlib.Path(__file__).parent / "../../../data/log/serv_cfg.txt").resolve()
DEFAULT_SERVER_IP = "io1"
DEFAULT_SERVER_PORT = 20543
HEADER_SIZE = 4

class JobClient:
    def __init__(self, host=None, port=None):
        if os.path.isfile(SERVER_CONFIG_PATH):
            with open(SERVER_CONFIG_PATH) as fp:
                cfg_host, cfg_port = fp.read().split()
        else:
            cfg_host, cfg_port = None, None

        if host is not None:
            self.host = host
        else:
            self.host = cfg_host if cfg_host is not None else DEFAULT_SERVER_IP
        
        if port is not None:
            self.port = port
        else:
            self.port = int(cfg_port) if cfg_port is not None else DEFAULT_SERVER_PORT
    
    def submit_batch(self, batch_id, tasks, timeout=120, ncore=4):
        payload = {
            "batch_id": batch_id,
            "tasks": [{"id": t["id"], "code": t["code"]} for t in tasks],
            "timeout": timeout,
            "ncore": ncore
        }
        
        with socket.socket() as s:
            s.connect((self.host, self.port))

            data = json.dumps(payload).encode()
            header = len(data).to_bytes(HEADER_SIZE, 'big')
            s.sendall(header + data)
            
            response = b''
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                response += chunk
            
            response = response.decode().replace('\x00', '')
            try:
                return json.loads(response)
            except Exception as e:
                print(f"Error decoding response: {e}")
                return None

if __name__ == "__main__":
    client = JobClient()
    print(f"Connecting to server at {client.host}:{client.port}")

    # 1: 
    print("\n=== 1:  ===")
    tasks1 = []
    for i in range(8):
        code = f'print("hello world {i}")'
        tasks1.append({"id": f"basic_task_{i}", "code": code})

    batch_id1 = f"test_basic_{int(time.time())}"
    print(f"Submitting batch {batch_id1} with {len(tasks1)} tasks...")

    result1 = client.submit_batch(batch_id1, tasks1, timeout=30, ncore=1)

    if result1 and "results" in result1:
        print(f"  {len(result1['results'])} ")
        for r in result1["results"][:3]:  #  3 
            print(f"   {r['id']}: {r['stdout'].strip()}")
    else:
        print(" ")

    # 2: 
    print("\n=== 2:  ===")
    tasks2 = [
        {"id": "zero_division", "code": "print('Starting'); result = 1/0; print(result)"},
        {"id": "import_error", "code": "import nonexistent_module"},
        {"id": "syntax_error", "code": "print('hello')  # missing quote"}
    ]

    batch_id2 = f"test_error_{int(time.time())}"
    print(f"Submitting error test batch {batch_id2}...")

    result2 = client.submit_batch(batch_id2, tasks2, timeout=30, ncore=1)

    if result2 and "results" in result2:
        print(f"  {len(result2['results'])} ")
        for r in result2["results"]:
            print(f"   {r['id']}: {r['stdout'][:100]}...")
    else:
        print(" ")

    # 3: 
    print("\n=== 3:  ===")
    tasks3 = [
        {"id": "sleep_task", "code": "import time; print('Starting sleep'); time.sleep(10); print('Should not see this')"}
    ]

    batch_id3 = f"test_timeout_{int(time.time())}"
    print(f"Submitting timeout test batch {batch_id3} (timeout=3s)...")

    result3 = client.submit_batch(batch_id3, tasks3, timeout=3, ncore=1)

    if result3 and "results" in result3:
        print(f" ")
        r = result3["results"][0]
        print(f"   {r['id']}: {r['stdout'][:150]}...")
    else:
        print(" ")

    # 4: 
    print("\n=== 4:  ===")
    for ncore in [1, 2, 4]:
        tasks4 = [
            {"id": f"cpu_test_{ncore}", "code": f'''
import os
import multiprocessing
print(f"Process PID: {{os.getpid()}}")
print(f"Configured cores: {ncore}")
print(f"Available CPUs: {{multiprocessing.cpu_count()}}")
print("Multi-core test completed successfully")
'''}
        ]

        batch_id4 = f"test_cpu_{ncore}_{int(time.time())}"
        print(f"Submitting CPU core test (ncore={ncore})...")

        result4 = client.submit_batch(batch_id4, tasks4, timeout=30, ncore=ncore)

        if result4 and "results" in result4:
            print(f" ncore={ncore}: {result4['results'][0]['stdout'].strip()}")
        else:
            print(f" ncore={ncore}: ")

    # 5: 
    print("\n=== 5:  ===")
    tasks5 = []
    for i in range(32):  # 
        code = f'''
import time
import random
delay = random.uniform(0.1, 0.5)
time.sleep(delay)
print("Concurrent task {i} completed after " + str(delay) + "s")
'''
        tasks5.append({"id": f"concurrent_task_{i}", "code": code})

    batch_id5 = f"test_concurrent_{int(time.time())}"
    print(f"Submitting concurrent batch {batch_id5} with {len(tasks5)} tasks...")

    result5 = client.submit_batch(batch_id5, tasks5, timeout=60, ncore=2)

    if result5 and "results" in result5:
        print(f"  {len(result5['results'])} ")
        success_count = sum(1 for r in result5["results"] if "completed" in r['stdout'])
        print(f"  : {success_count}/{len(tasks5)}")

        # 
        for i, r in enumerate(result5["results"][:3]):
            print(f"   {i+1}: {r['stdout'].strip()}")
    else:
        print(" ")

    print("\n===  ===")
    print(f": {client.host}:{client.port}")
    print("")