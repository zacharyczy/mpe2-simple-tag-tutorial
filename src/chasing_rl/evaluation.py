"""Independent fixed-seed evaluation with no learning or training RNG effects."""
from contextlib import contextmanager
import random
import numpy as np
import torch
from chasing_rl.envs.pursuit_env import PursuitEnv

@contextmanager
def preserve_rng():
    py=random.getstate(); np_state=np.random.get_state(); cpu=torch.get_rng_state()
    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try: yield
    finally:
        random.setstate(py);np.random.set_state(np_state);torch.set_rng_state(cpu)
        if cuda is not None: torch.cuda.set_rng_state_all(cuda)

def evaluate(policy,env_factory,seeds,render=False,gif_path=None):
    records=[];frames=[]
    was_training=policy.training
    with preserve_rng():
        env=env_factory()
        try:
            policy.eval()
            for episode,seed in enumerate(seeds):
                frame=env.reset(seed=int(seed),opponent_seed=int(seed)+1_000_000)
                total=0.
                if gif_path and episode==0: frames.append(env.render())
                while True:
                    actions,_=policy.act(frame.obs,deterministic=True)
                    transition=env.step(actions)
                    total+=transition.reward
                    frame=transition.next_frame
                    if gif_path and episode==0:frames.append(env.render())
                    if render:env.render()
                    if transition.done:
                        records.append(dict(seed=int(seed),success=transition.success,
                                            steps=transition.info['steps'],team_return=round(total,8)))
                        break
        finally:
            env.close()
            policy.train(was_training)
    if not records: raise ValueError('evaluation seeds must not be empty')
    if gif_path:
        import imageio.v2 as imageio
        from pathlib import Path
        Path(gif_path).parent.mkdir(parents=True,exist_ok=True)
        imageio.mimsave(gif_path,frames,duration=100,loop=0)
    return dict(episodes=len(records),success_rate=float(np.mean([r['success'] for r in records])),
                mean_steps_including_failures=float(np.mean([r['steps'] for r in records])),
                mean_team_return=float(np.mean([r['team_return'] for r in records])),records=records)
