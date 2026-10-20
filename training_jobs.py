"""In-process background training jobs with progress/log/cancel state."""
from threading import Thread, Event, Lock
from uuid import uuid4
from pathlib import Path
from datetime import datetime, timezone
import traceback
import re
from config import PLANT
from experiment_engine import plant_to_dict, resolve_plant_preset
from policy_registry import MODEL_DIR, register_policy
from sac_training import train_sac

_JOBS={}; _LOCK=Lock()
def _public(job):
    return {k:v for k,v in job.items() if k not in ("thread","stop_event")}
def list_jobs():
    with _LOCK: return [_public(j) for j in _JOBS.values()]
def get_job(job_id):
    with _LOCK:
        if job_id not in _JOBS: raise KeyError(job_id)
        return _public(_JOBS[job_id])
def cancel_job(job_id):
    with _LOCK:
        if job_id not in _JOBS: raise KeyError(job_id)
        job=_JOBS[job_id]
        if job["status"] not in ("running", "queued"): return _public(job)
        job["stop_event"].set(); job["logs"].append("Cancellation requested; training will stop at the next SAC callback.")
        return _public(job)
def start_job(payload):
    job_id=uuid4().hex[:12]; stop=Event()
    mode=payload.get("training_mode","individual")
    if mode not in ("individual","generalized"): raise ValueError("training_mode must be individual or generalized")
    # Validate the plant preset/variation request before starting a background job.
    resolve_plant_preset(payload.get("plant_preset", "paper_nominal"), payload.get("variations", {}))
    name=str(payload.get("name") or f"{mode}_{job_id}").strip()[:80]
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or mode
    model_path=MODEL_DIR/f"{job_id}_{safe_name}"
    job={"id":job_id,"name":name,"status":"queued","progress":0.0,"timesteps":0,
         "total_timesteps":int(payload.get("total_timesteps",50000)),"logs":["Job queued"],
         "created_at":datetime.now(timezone.utc).isoformat(),"error":None,"policy_id":None,
         "thread":None,"stop_event":stop}
    if job["total_timesteps"]<1 or job["total_timesteps"]>5_000_000: raise ValueError("total_timesteps must be between 1 and 5,000,000")
    with _LOCK: _JOBS[job_id]=job
    def progress(frac, steps, total, message):
        with _LOCK:
            job["progress"]=float(frac); job["timesteps"]=int(steps)
            if not job["logs"] or job["logs"][-1]!=message: job["logs"].append(message)
            job["logs"]=job["logs"][-100:]
    def worker():
        try:
            with _LOCK: job["status"]="running"; job["logs"].append("Training started")
            training_plant, training_variations = resolve_plant_preset(payload.get("plant_preset", "paper_nominal"), payload.get("variations", {}))
            train_sac(total_timesteps=job["total_timesteps"], seed=int(payload.get("seed",42)),
                model_path=str(model_path), plant_config=training_plant,
                generalized=(mode=="generalized"), plant_variation_pct=float(payload.get("plant_variation_pct",10)),
                force_model_config=payload.get("cutting_force",{"kind":"sinusoidal"}),
                noise_config=payload.get("noise",{}), on_progress=progress, stop_event=stop)
            cancelled=stop.is_set()
            policy_id=job_id
            metadata={"id":policy_id,"name":name,"model_path":(model_path.relative_to(Path(__file__).resolve().parent).as_posix()+".zip"),
                "training_mode":mode,"plant":plant_to_dict(training_plant),"plant_preset":payload.get("plant_preset", "paper_nominal"),
                "training_variations":training_variations,"plant_variation_pct":float(payload.get("plant_variation_pct",10)),
                "cutting_force":payload.get("cutting_force",{"kind":"sinusoidal"}),"noise":payload.get("noise",{}),
                "seed":int(payload.get("seed",42)),"total_timesteps":job["timesteps"],"requested_timesteps":job["total_timesteps"],
                "partial_checkpoint":cancelled,"observation_dimension":3*8,"action_dimension":1}
            register_policy(metadata)
            with _LOCK:
                job["status"]="cancelled" if cancelled else "completed"; job["policy_id"]=policy_id
                job["progress"]=min(job["timesteps"]/max(job["total_timesteps"],1),1.0); job["logs"].append("Partial checkpoint saved after cancellation." if cancelled else "Training completed and policy registered.")
        except Exception as exc:
            with _LOCK: job["status"]="failed"; job["error"]=str(exc); job["logs"].append(traceback.format_exc()[-3000:])
    th=Thread(target=worker,daemon=True,name=f"sac-train-{job_id}"); job["thread"]=th; th.start()
    return get_job(job_id)
