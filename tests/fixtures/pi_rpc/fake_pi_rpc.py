#!/usr/bin/env python3
"""Adversarial subprocess fixture for the strict Pi JSONL client tests."""
from __future__ import annotations

import json
import subprocess
import sys
import time


mode = sys.argv[1]
if mode == "malformed":
    sys.stdout.write("not-json\n")
    sys.stdout.flush()
    time.sleep(1)
elif mode == "partial":
    sys.stdout.write('{"type":"event"')
    sys.stdout.flush()
elif mode == "hang":
    for _line in sys.stdin:
        time.sleep(60)
elif mode == "stderr":
    sys.stderr.write("x" * 100_000 + " sk-" + "s" * 40)
    sys.stderr.flush()
    time.sleep(0.2)
elif mode == "child":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    print(json.dumps({"type": "child", "pid": child.pid}), flush=True)
    for _line in sys.stdin:
        time.sleep(60)
elif mode == "child_exit":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    print(json.dumps({"type": "child", "pid": child.pid}), flush=True)
elif mode == "echo":
    for line in sys.stdin:
        request = json.loads(line)
        print(json.dumps({"type": "agent_start", "request": request["id"]}), flush=True)
        print(json.dumps({
            "type": "response",
            "id": request["id"],
            "command": request["type"],
            "success": True,
            "data": {},
        }), flush=True)
