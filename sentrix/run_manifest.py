"""Record software/config/weight fingerprints for a reproducible experiment."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from .config import Settings


def sha256(path):
    if not path or not os.path.isfile(path):return None
    digest=hashlib.sha256()
    with open(path,"rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("config");p.add_argument("--out",default="reports/run_manifest.json");a=p.parse_args()
    settings=Settings.from_json(a.config)
    packages={}
    for name in ("ultralytics","opencv-python","numpy","pandas","streamlit","torch","tensorflow"):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    result={"python":platform.python_version(),"platform":platform.platform(),"config_path":a.config,"config_sha256":sha256(a.config),
        "detector_weights":settings.model,"detector_weights_sha256":sha256(settings.model),"pose_enabled":settings.pose_enabled,
        "pose_weights_sha256":sha256(settings.pose_model) if settings.pose_enabled else None,"packages":packages,
        "hardware_device":"Record actual CPU/GPU and camera resolution in the run notes before reporting throughput."}
    os.makedirs(os.path.dirname(a.out) or ".",exist_ok=True)
    with open(a.out,"w",encoding="utf-8") as handle:json.dump(result,handle,indent=2)
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
