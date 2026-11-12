"""有限场地的行为检查。运行 python -m unittest discover -s tests -v。"""

import argparse
import sys
import unittest
from pathlib import Path

import numpy as np
from pettingzoo.test import parallel_api_test

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simple_tag_policy_demo import flee_action
from square_tag_env import SquareWorld, arena_half_size, parallel_env
from mpe2._mpe_utils.core import Agent


class SquareTagTests(unittest.TestCase):
    def test_wall_retains_tangential_velocity(self):
        world = SquareWorld(1.5)
        agent = Agent()
        agent.size = 0.05
        agent.state.p_pos = np.array([1.44, 0.])
        agent.state.p_vel = np.array([1., 0.4])
        world.agents = [agent]
        world.integrate_state([None])
        np.testing.assert_allclose(agent.state.p_pos, [1.45, 0.04])
        np.testing.assert_allclose(agent.state.p_vel, [0., 0.3])

    def test_four_walls_and_corners(self):
        for x_sign in (-1, 1):
            for y_sign in (-1, 1):
                world = SquareWorld(1.5)
                agent = Agent()
                agent.size = 0.075
                agent.state.p_pos = np.array([x_sign * 1.42, y_sign * 1.42])
                agent.state.p_vel = np.array([x_sign * 1., y_sign * 1.])
                world.agents = [agent]
                world.integrate_state([None])
                np.testing.assert_allclose(agent.state.p_pos, [x_sign * 1.425, y_sign * 1.425])
                np.testing.assert_array_equal(agent.state.p_vel, [0., 0.])

    def test_inward_motion_is_allowed(self):
        world = SquareWorld(1.5)
        agent = Agent()
        agent.state.p_pos = np.array([1.45, 0.])
        agent.state.p_vel = np.array([-1., 0.])
        world.agents = [agent]
        world.integrate_state([None])
        self.assertLess(agent.state.p_pos[0], 1.45)
        self.assertLess(agent.state.p_vel[0], 0.)

    @staticmethod
    def observation(position, nearest, velocity=(0., 0.)):
        obs = np.zeros(14)
        obs[:2], obs[2:4] = velocity, position
        obs[8:14] = np.concatenate([nearest, [8., 8.], [-8., -8.]])
        return obs

    def test_flee_open_space_and_unbounded(self):
        obs = self.observation([0., 0.], [-0.5, -0.2])
        self.assertEqual(flee_action(obs, 1.5), 2)
        obs[2] = 1.44
        self.assertEqual(flee_action(obs, None), 2)

    def test_flee_chooses_second_best_tangent(self):
        # 首选向右已被墙阻挡；向上比向下、向左更远离威胁。
        obs = self.observation([1.44, 0.], [-0.5, -0.2])
        self.assertEqual(flee_action(obs, 1.5), 4)

    def test_flee_uses_velocity_before_wall(self):
        obs = self.observation([1.2, 0.], [-0.5, -0.2], [1.3, 0.])
        self.assertEqual(flee_action(obs, 1.5), 4)
        obs[:2] = 0.
        self.assertEqual(flee_action(obs, 1.5), 2)

    def test_corner_always_returns_feasible_action(self):
        obs = self.observation([1.44, 1.44], [-0.5, -0.2])
        self.assertEqual(flee_action(obs, 1.5), 3)

    def test_reward_and_observation_follow_wall_projection(self):
        env = parallel_env(max_cycles=10)
        try:
            env.reset(seed=1)
            raw = env.unwrapped
            pursuer, _, _, prey = raw.world.agents
            for i, agent in enumerate(raw.world.agents):
                agent.state.p_pos = np.array([-1., -1. + i * 0.3])
                agent.state.p_vel = np.zeros(2)
            for i, landmark in enumerate(raw.world.landmarks):
                landmark.state.p_pos = np.array([-1., 0.6 + i * 0.4])
            pursuer.state.p_pos = np.array([1.35, 0.])
            pursuer.state.p_vel = np.zeros(2)
            prey.state.p_pos = np.array([1.45, 0.])
            prey.state.p_vel = np.array([1.3, 0.])
            obs, rewards, terminated, _, _ = env.step({a: 0 for a in env.agents})
            # 校正后的状态必须用于捕获、奖励与观测，不能只修画面。
            np.testing.assert_allclose(obs[prey.name][2:4], prey.state.p_pos, atol=1e-6)
            self.assertTrue(all(terminated.values()))
            self.assertEqual(rewards[pursuer.name], 10.)
            self.assertLessEqual(prey.state.p_pos[0], 1.45)
        finally:
            env.close()

    def test_seed_shapes_and_long_rollout(self):
        env = parallel_env(max_cycles=150)
        try:
            first, _ = env.reset(seed=42)
            again, _ = env.reset(seed=42)
            for name in first:
                np.testing.assert_array_equal(first[name], again[name])
                self.assertEqual(first[name].shape, (14,) if name == "agent_0" else (16,))
                env.action_space(name).seed(42)
            for _ in range(1000):
                env.step({name: env.action_space(name).sample() for name in env.agents})
                for entity in env.unwrapped.world.entities:
                    self.assertTrue(np.all(np.abs(entity.state.p_pos) + entity.size <= 1.5 + 1e-12))
                if not env.agents:
                    env.reset()
        finally:
            env.close()

    def test_parallel_api_and_render(self):
        env = parallel_env(render_mode="rgb_array")
        try:
            parallel_api_test(env, num_cycles=50)
            env.reset(seed=42)
            frame = env.render()
            self.assertEqual(frame.shape, (700, 700, 3))
            self.assertEqual(frame.dtype, np.uint8)
            self.assertTrue(np.all(frame[50, 100] == [65, 65, 65]))
        finally:
            env.close()

    def test_invalid_arena_size(self):
        for size in (0, -1, 1, "nan", "inf", "bad"):
            with self.assertRaises(argparse.ArgumentTypeError):
                arena_half_size(size)


if __name__ == "__main__":
    unittest.main()
