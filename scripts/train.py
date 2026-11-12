"""Train the shared feed-forward MAPPO baseline."""
import argparse
from pathlib import Path
from chasing_rl.runner import read_config, train

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,help='Override seed for independent training runs')
    a=p.parse_args();config=read_config(a.config)
    if a.seed is not None:config['seed']=a.seed
    train(config,a.output)

if __name__=='__main__':main()
