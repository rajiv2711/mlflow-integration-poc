# caller/call_train.py  (or server_caller.py)

import requests, webbrowser, time, sys, pprint

payload = {"alpha": 0.001, "epochs": 60}

resp = requests.post("http://localhost:8001/train", json=payload)

# Bail out early if the trainer reported an error
if resp.status_code != 200:
    print("Trainer returned HTTP", resp.status_code)
    print(resp.text)          # already JSON thanks to our try/except in server.py
    sys.exit(1)

data = resp.json()            # <- convert Response ➞ dict
print("MLflow run:", data)    # {'experiment_id': '1', 'run_id': 'abc123...'}

time.sleep(1)                 # give MLflow UI a split-second to refresh
url = f"http://localhost:5000/#/experiments/{data['experiment_id']}/runs/{data['run_id']}"
print("Opening", url)
webbrowser.open(url)
