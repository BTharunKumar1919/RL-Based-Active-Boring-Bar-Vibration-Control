"""File-backed registry for SAC checkpoints and their training metadata."""
import json
from pathlib import Path
from datetime import datetime, timezone
from threading import Lock

ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "results" / "policy_registry.json"
MODEL_DIR = ROOT / "results" / "policies"
_REGISTRY_LOCK = Lock()

def list_policies():
    data=[]
    if REGISTRY_PATH.exists():
        try: data=json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): data=[]
    if not isinstance(data, list): data=[]
    # Surface the bundled current-driven checkpoint even before the user trains
    # a policy through the new registry workflow.
    legacy_path = ROOT / "results" / "sac_vibration_model_7mode.zip"
    if legacy_path.is_file() and not any(x.get("id") == "bundled_7mode" for x in data):
        data.append({"id":"bundled_7mode", "name":"Bundled 7-mode SAC checkpoint", "model_path":str(legacy_path),
            "training_mode":"individual", "plant":None, "cutting_force":{"kind":"sinusoidal"},
            "noise":{}, "seed":None, "total_timesteps":None, "observation_dimension":24, "action_dimension":1,
            "legacy_checkpoint":True})
    items=[]
    for item in data:
        p=Path(item.get("model_path",""))
        if not p.is_absolute(): p=ROOT/p
        item=dict(item); item["model_exists"]=p.is_file(); item["model_path"]=str(p)
        items.append(item)
    return items

def get_policy(policy_id):
    for item in list_policies():
        if item.get("id")==policy_id: return item
    raise KeyError(f"Unknown policy id: {policy_id}")

def register_policy(item):
    with _REGISTRY_LOCK:
        REGISTRY_PATH.parent.mkdir(parents=True,exist_ok=True)
        data=[]
        if REGISTRY_PATH.exists():
            try: data=json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
            except (OSError,json.JSONDecodeError): data=[]
        item=dict(item); item.setdefault("created_at",datetime.now(timezone.utc).isoformat())
        data=[x for x in data if x.get("id")!=item.get("id")]; data.append(item)
        REGISTRY_PATH.write_text(json.dumps(data,indent=2),encoding="utf-8")
        return item
