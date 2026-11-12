"""Evaluate a trusted checkpoint; never update parameters."""
import argparse,json
from pathlib import Path
from chasing_rl.checkpoint import load_checkpoint
from chasing_rl.runner import env_factory
from chasing_rl.evaluation import evaluate

def main():
    p=argparse.ArgumentParser(__doc__)
    choice=p.add_mutually_exclusive_group(required=True)
    choice.add_argument('--checkpoint',type=Path)
    choice.add_argument('--rule',choices=['chase'])
    p.add_argument('--config',type=Path,default=Path('configs/mappo_baseline.json'))
    p.add_argument('--opponent',choices=['flee','random','stay'],default='flee')
    p.add_argument('--episodes',type=int,default=200)
    p.add_argument('--seed-start',type=int,required=True,help='Explicit development/validation/test seed range')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.episodes<1:p.error('episodes must be positive')
    if a.checkpoint:
        policy,data=load_checkpoint(a.checkpoint)
    else:
        from chasing_rl.policies.rules import RulePolicy
        from chasing_rl.runner import read_config
        policy=RulePolicy()
        data={'config':read_config(a.config),'env_steps':0}
    result=evaluate(policy,lambda:env_factory(data['config'],a.opponent),range(a.seed_start,a.seed_start+a.episodes))
    result.update(checkpoint=str(a.checkpoint) if a.checkpoint else None,rule=a.rule,opponent=a.opponent,env_steps=data['env_steps'],deterministic=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))

if __name__=='__main__':main()
