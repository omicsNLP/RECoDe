import os
import requests


RECORD_ID = "19050554"
OUTPUT_DIR = f"../../data/annotation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

api_url = f"https://zenodo.org/api/records/{RECORD_ID}"
resp = requests.get(api_url, timeout=60)
resp.raise_for_status()
data = resp.json()

files = data.get("files", [])
if not files:
    print("No files found.")
    raise SystemExit(1)

print(f"Found {len(files)} files")

for f in files:
    filename = f["key"]
    url = f["links"]["self"]

    out_path = os.path.join(OUTPUT_DIR, filename)
    print(f"Downloading {filename}...")

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as wf:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    wf.write(chunk)

print("Done.")