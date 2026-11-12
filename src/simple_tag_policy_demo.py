"""教学示例：将双方的决策函数接入 Simple Tag，不含强化学习训练。

默认使用边长 3 的正方形硬边界；--unbounded 可运行原始无墙版本。
仅适用于下方固定的 3 追捕者、1 逃逸者、2 障碍物、全实体观测配置。
先读 README，再按 direction_to_action -> select_action -> main 的顺序阅读。
"""

import argparse

import numpy as np
from mpe2 import simple_tag_v3

# 只借用两个辅助函数；不会运行 simple_tag_demo.py 中的 main()。
from simple_tag_demo import capture_success, positive_int
from square_tag_env import DEFAULT_ARENA_HALF_SIZE, arena_half_size, parallel_env


def direction_to_action(direction):
    """把期望方向 [dx, dy] 转成一个离散动作，不同时走两个方向。"""
    dx, dy = direction
    if max(abs(dx), abs(dy)) < 1e-8:
        return 0
    # 横纵差相同时优先横向；这是本示例的约定，不是库的要求。
    if abs(dx) >= abs(dy):
        return 2 if dx > 0 else 1
    return 4 if dy > 0 else 3


def flee_action(observation, half_size):
    """按远离最近追捕者的投影排序，剔除朝近墙外运动的动作。

    只读自己的观测和公共场地配置，不读取环境内部状态。
    这是边界感知规则，不包含障碍物规划或对追捕者的运动预测。
    """
    relative_positions = observation[8:14].reshape(3, 2)
    nearest = relative_positions[int(np.argmin(np.linalg.norm(relative_positions, axis=1)))]
    away = -nearest
    if half_size is None:
        return direction_to_action(away)  # --unbounded 复现原规则
    if np.linalg.norm(away) < 1e-8:
        return 0
    position, velocity = observation[2:4], observation[0:2]
    # 0.05 是固定配置下逃逸者半径；0.08 留出转向余量。
    # 额外考虑当前朝墙速度在两个物理步长 (dt=0.1) 内的位移。
    limit = half_size - 0.05
    directions = {1: np.array([-1., 0.]), 2: np.array([1., 0.]),
                  3: np.array([0., -1.]), 4: np.array([0., 1.])}
    candidates = []
    for action, direction in directions.items():
        axis = int(np.argmax(np.abs(direction)))
        sign = direction[axis]
        clearance = limit - sign * position[axis]
        margin = 0.08 + max(0.0, sign * velocity[axis]) * 0.2
        if clearance <= margin:
            continue
        candidates.append(action)
    # 同分按动作编号取先者，与原规则一样横向优先；沿墙两向同分时向下优先。
    return max(candidates, key=lambda a: float(np.dot(directions[a], away))) if candidates else 0


def select_action(agent, observation, action_space, pursuer_policy, evader_policy,
                  half_size=DEFAULT_ARENA_HALF_SIZE):
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
        # 根据自己的位置、速度、最近威胁和公共边界选择可行逃离方向。
        return flee_action(observation, half_size)
    raise ValueError(f"Unsupported policy {policy!r} for {agent!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pursuers", choices=("random", "chase", "stay"), default="random")
    parser.add_argument("--evader", choices=("random", "flee", "stay"), default="random")
    parser.add_argument("--episodes", type=positive_int, default=5)
    parser.add_argument("--max-cycles", type=positive_int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", choices=("human", "none"), default="human")
    parser.add_argument("--arena-half-size", type=arena_half_size, default=DEFAULT_ARENA_HALF_SIZE,
                        help="Square half side length (default: 1.5, full side: 3)")
    parser.add_argument("--unbounded", action="store_true", help="Use original MPE2 and flee rule")
    parser.add_argument("--inspect", action="store_true", help="Print first episode's first transition")
    args = parser.parse_args()

    # 原随机示例不变；此教学示例默认使用项目中的有限场地环境。
    half_size = None if args.unbounded else args.arena_half_size
    if args.unbounded:
        env = simple_tag_v3.parallel_env(
            num_good=1, num_adversaries=3, num_obstacles=2,
            max_cycles=args.max_cycles, continuous_actions=False,
            terminate_on_success=True, curriculum=False,
            num_agent_neighbors=None, num_landmark_neighbors=None,
            render_mode="human" if args.render == "human" else None,
        )
    else:
        env = parallel_env(arena_half_size=half_size, max_cycles=args.max_cycles,
                           render_mode="human" if args.render == "human" else None)
    arena_mode = "unbounded" if args.unbounded else "bounded"
    print(f"arena={arena_mode}, arena_half_size={half_size}, "
          f"max_cycles={args.max_cycles}, seed={args.seed}")
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
                        args.pursuers, args.evader, half_size,
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
