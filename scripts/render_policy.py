"""Render a trained policy without learning; optionally save one development GIF."""
import argparse
from pathlib import Path
from chasing_rl.checkpoint import load_checkpoint
from chasing_rl.runner import env_factory
from chasing_rl.evaluation import evaluate

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--seed',type=int,default=1000000)
    p.add_argument('--gif',type=Path)
    a=p.parse_args();policy,data=load_checkpoint(a.checkpoint)
    mode='rgb_array' if a.gif else 'human'
    print(evaluate(policy,lambda:env_factory(data['config'],render_mode=mode),[a.seed],
                   render=not bool(a.gif),gif_path=a.gif))

if __name__=='__main__':main()
