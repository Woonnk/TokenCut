"""Illustrative audit fixture; not an executable agent."""

import json
import subprocess


def run_agent(client, repository_file, history):
    whole_file = repository_file.read_text()
    test_run = subprocess.run(["python", "-m", "unittest"], capture_output=True, text=True)
    payload = json.dumps({"file": whole_file, "logs": test_run.stdout}, indent=4)
    history.append({"role": "user", "content": payload})
    for _ in range(20):
        response = client.chat.completions.create(messages=history)
        history.append(response)
    return history
