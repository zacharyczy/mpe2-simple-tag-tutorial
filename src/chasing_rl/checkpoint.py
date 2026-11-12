"""Trusted local checkpoints for evaluation; exact training resume is not supported."""
import json
import hashlib
from pathlib import Path
import subprocess
import importlib.metadata
import torch
from chasing_rl.policies.actor_critic import Policy

PROJECT_ROOT=Path(__file__).resolve().parents[2]
UPSTREAM_REVISION='de66d7a4b23fac2513f56f96f73b3f5cb96695ac'

def provenance():
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT_ROOT,text=True).strip()
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=PROJECT_ROOT,text=True).strip())
    except (OSError,subprocess.CalledProcessError): commit,dirty='unknown',True
    tracked_sources=sorted((PROJECT_ROOT/'src').rglob('*.py'))
    hashes={str(p.relative_to(PROJECT_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked_sources}
    return dict(source_sha256=hashes,project_commit=commit,working_tree_dirty=dirty,upstream_commit=UPSTREAM_REVISION,
                versions={k:importlib.metadata.version(k) for k in ['mpe2','torch','numpy']})

def save_checkpoint(path,policy,config,steps,source=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=dict(format_version=1,policy=policy.state_dict(),config=config,env_steps=steps,
              provenance=source or provenance(),normalization=None,exact_resume_supported=False)
    temp=path.with_suffix('.tmp')
    torch.save(data,temp)
    temp.replace(path)

def load_checkpoint(path,device='cpu'):
    # Checkpoints contain tensors and primitive metadata only. Never need pickle objects.
    data=torch.load(path,map_location=device,weights_only=True)
    if data['format_version']!=1: raise ValueError('unsupported checkpoint')
    policy=Policy(data['config']['hidden_size'],device)
    policy.load_state_dict(data['policy'])
    policy.eval()
    return policy,data
