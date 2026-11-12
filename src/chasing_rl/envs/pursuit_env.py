"""训练适配：不修改 SquareTagEnv 物理与原演示。"""
from dataclasses import dataclass
import numpy as np
from square_tag_env import parallel_env
from chasing_rl.policies.rules import flee_action

PURSUERS = ("adversary_0", "adversary_1", "adversary_2")

@dataclass
class Frame:
    obs: np.ndarray
    state: np.ndarray
    info: dict

@dataclass
class Transition:
    next_frame: Frame
    reward: float
    success: bool
    timeout: bool
    info: dict

    @property
    def done(self):
        return self.success or self.timeout

class PursuitEnv:
    def __init__(self, max_steps=100, arena_half_size=1.5, opponent="flee", render_mode=None):
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if opponent not in ("flee", "random", "stay"):
            raise ValueError("unknown opponent")
        self.max_steps, self.half_size, self.opponent = max_steps, arena_half_size, opponent
        self.backend = parallel_env(arena_half_size=arena_half_size, max_cycles=max_steps,
                                    render_mode=render_mode)
        self.raw = self.backend.unwrapped
        self.steps, self.done = 0, True
        self.rng = np.random.default_rng(0)

    def geometry(self):
        w = self.raw.world
        return (np.stack([a.state.p_pos.copy() for a in w.agents]),
                np.stack([a.state.p_vel.copy() for a in w.agents]),
                np.stack([b.state.p_pos.copy() for b in w.landmarks]))

    def captured(self):
        w = self.raw.world
        return any(self.raw.scenario.is_collision(a, w.agents[3]) for a in w.agents[:3])

    def frame(self):
        tau = (self.max_steps - self.steps) / self.max_steps
        obs = np.stack([np.r_[self.raw.observe(a), tau] for a in PURSUERS]).astype(np.float32)
        pos, vel, obstacles = self.geometry()
        state = np.r_[pos.ravel(), vel.ravel(), obstacles.ravel(), tau].astype(np.float32)
        if obs.shape != (3,17) or state.shape != (21,):
            raise RuntimeError("environment layout changed; audit before training")
        if not np.isfinite(obs).all() or not np.isfinite(state).all():
            raise FloatingPointError('non-finite environment observation/state')
        return Frame(obs, state, {"steps": self.steps})

    def reset(self, seed=None, opponent_seed=None):
        self.rng = np.random.default_rng(opponent_seed)
        for attempt in range(1,101):
            self.backend.reset(seed=seed if attempt==1 else None)
            if not self.captured():
                self.steps, self.done = 0, False
                f = self.frame()
                f.info.update(reset_attempts=attempt)
                return f
        raise RuntimeError("failed to sample a non-captured initial state")

    def step(self, pursuer_actions):
        if self.done:
            raise RuntimeError("reset before step")
        a = np.asarray(pursuer_actions)
        if a.shape != (3,) or a.dtype.kind not in "iu" or np.any((a<0)|(a>4)):
            raise ValueError("expected three integer actions in 0..4")
        obs_e = self.raw.observe("agent_0")
        evader = (flee_action(obs_e,self.half_size) if self.opponent=="flee" else
                  int(self.rng.integers(5)) if self.opponent=="random" else 0)
        joint = dict(zip(PURSUERS,map(int,a)))
        joint['agent_0'] = evader
        _, rewards, terms, truncs, _ = self.backend.step(joint)
        self.steps += 1
        success = bool(self.captured())
        timeout = self.steps >= self.max_steps and not success
        self.done = success or timeout
        if not self.done and (any(terms.values()) or any(truncs.values())):
            raise RuntimeError("unexpected backend termination")
        return Transition(self.frame(), float(success)-.01, success, timeout,
                          {"steps":self.steps,"evader_action":evader,"native_rewards":rewards,
                           "native_terminated":terms,"native_truncated":truncs})

    def render(self):
        return self.backend.render()

    def close(self):
        self.backend.close()
