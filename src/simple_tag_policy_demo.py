"""教学示例：将双方的决策函数接入 Simple Tag，不含强化学习训练。

仅适用于下方固定的 3 追捕者、1 逃逸者、2 障碍物、全实体观测配置。
先读 README，再按 direction_to_action -> select_action -> main 的顺序阅读。
"""

import argparse

import numpy as np
from mpe2 import simple_tag_v3

# 只借用两个辅助函数；不会运行 simple_tag_demo.py 中的 main()。
from simple_tag_demo import capture_success, positive_int


def direction_to_action(direction):
    """把期望方向 [dx, dy] 转成一个离散动作，不同时走两个方向。"""
    dx, dy = direction
    if max(abs(dx), abs(dy)) < 1e-8:
        return 0
    # 横纵差相同时优先横向；这是本示例的约定，不是库的要求。
    if abs(dx) >= abs(dy):
        return 2 if dx > 0 else 1
    return 4 if dy > 0 else 3


def select_action(agent, observation, action_space, pursuer_policy, evader_policy):
    """决策入口：只使用当前智能体自己的 observation，返回一个动作编号。"""
    # agent 是名称字符串，例如 "adversary_0"；observation 是它当前的数值向量。
    is_pursuer = agent.startswith("adversary_")
    policy = pursuer_policy if is_pursuer else evader_policy
    expected_shape = (16,) if is_pursuer else (14,)
    if observation.shape != expected_shape:
        raise ValueError("Observation layout changed; update the policy slices first.")

    if policy == "random":
        return int(action_space.sample())
    if policy == "stay":
        # 0 表示不施加主动运动控制；惯性、阻尼和碰撞仍由环境计算。
        return 0
    if is_pursuer and policy == "chase":
        # 在这里改追捕方规则，运行时用 --pursuers chase 才会进入该分支。
        # 三个追捕者的观测中，唯一逃逸者的相对位置都在 [12:14]。
        prey_relative_position = observation[12:14]
        return direction_to_action(prey_relative_position)
    if not is_pursuer and policy == "flee":
        # 在这里改逃逸方规则，运行时用 --evader flee 才会进入该分支。
        # 逃逸者的 [8:14] 依次是三个追捕者相对于自己的二维位置。
        pursuer_relative_positions = observation[8:14].reshape(3, 2)
        distances = np.linalg.norm(pursuer_relative_positions, axis=1)
        nearest = pursuer_relative_positions[int(np.argmin(distances))]
        return direction_to_action(-nearest)
    raise ValueError(f"Unsupported policy {policy!r} for {agent!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pursuers", choices=("random", "chase", "stay"), default="random")
    parser.add_argument("--evader", choices=("random", "flee", "stay"), default="random")
    parser.add_argument("--episodes", type=positive_int, default=5)
    parser.add_argument("--max-cycles", type=positive_int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", choices=("human", "none"), default="human")
    parser.add_argument("--inspect", action="store_true", help="Print first episode's first transition")
    args = parser.parse_args()

    # 在这里配置实体数量、动作类型和观测范围；Agent 对象由库内部创建。
    # continuous_actions=False 会让库创建 Discrete(5)，不必自己创建动作空间。
    env = simple_tag_v3.parallel_env(
        num_good=1, num_adversaries=3, num_obstacles=2,
        max_cycles=args.max_cycles, continuous_actions=False,
        terminate_on_success=True, curriculum=False,
        num_agent_neighbors=None, num_landmark_neighbors=None,
        render_mode="human" if args.render == "human" else None,
    )
    successes = 0
    total_steps = 0
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            # 观测由库计算并返回；在这里或下面的 step() 后可打印/断点查看。
            observations, infos = env.reset(seed=seed)
            for index, agent in enumerate(env.possible_agents):
                env.action_space(agent).seed(seed * len(env.possible_agents) + index)
            returns = dict.fromkeys(env.possible_agents, 0.0)
            steps = 0
            success = False
            while env.agents:
                # 1. 保存时刻 t 的观测，供双方策略读取。
                current_observations = observations
                # 2. 分别决策；这一过程中还没有调用 step，世界不会推进。
                actions = {
                    agent: select_action(
                        agent, current_observations[agent], env.action_space(agent),
                        args.pursuers, args.evader,
                    )
                    for agent in env.agents
                }
                # 3. 一次提交双方动作，得到 t+1 的观测和本步奖励。
                observations, rewards, terminated, truncated, infos = env.step(actions)
                steps += 1
                for agent, reward in rewards.items():
                    returns[agent] += float(reward)
                if args.inspect and episode == 0 and steps == 1:
                    for agent in current_observations:
                        print(f"observation_t[{agent}] = {current_observations[agent]}")
                    print("actions_t =", actions)
                    print("rewards_t =", dict(rewards))
                    print("terminated =", terminated, "truncated =", truncated)
                    print("observation_t_plus_1[adversary_0] =", observations["adversary_0"])
                success = capture_success(rewards, terminated)
                if success or all(truncated.values()):
                    break
            successes += int(success)
            total_steps += steps
            print(f"episode={episode + 1} seed={seed} steps={steps} "
                  f"success={success} returns={returns}")
    finally:
        env.close()
    print(f"pursuers={args.pursuers}, evader={args.evader}; no training")
    print(f"success_rate={successes / args.episodes:.1%}, "
          f"mean_steps_including_failures={total_steps / args.episodes:.2f}")


if __name__ == "__main__":
    main()
