"""MPE2 Simple Tag 随机动作测试；不含强化学习训练。

官方场景：https://mpe2.farama.org/environments/simple_tag/
使用 Parallel API，每步为所有智能体各执行一次动作。
"""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from mpe2 import simple_tag_v3


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def capture_success(rewards, terminated):
    # MPE2 1.1.1 在最后一步优先标记超时；默认奖励中追捕方
    # 仅在接触时获得正奖励，因此仍需把上限步发生的接触计为成功。
    return any(terminated.values()) or any(
        reward > 0 for agent, reward in rewards.items() if agent.startswith("adversary_")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=positive_int, default=5)
    parser.add_argument("--max-cycles", type=positive_int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", choices=("human", "none"), default="human")
    parser.add_argument("--output", type=Path, help="Optional JSON results path")
    args = parser.parse_args()

    env = simple_tag_v3.parallel_env(
        num_good=1,
        num_adversaries=3,
        num_obstacles=2,
        max_cycles=args.max_cycles,
        continuous_actions=False,
        terminate_on_success=True,
        render_mode="human" if args.render == "human" else None,
    )
    records = []
    try:
        print(f"MPE2 {version('mpe2')} | random actions | max_cycles={args.max_cycles}")
        for episode in range(args.episodes):
            seed = args.seed + episode
            observations, _ = env.reset(seed=seed)
            agents = list(env.possible_agents)
            # 环境初始位置和各智能体的随机动作分别设种子，以便复现。
            for index, agent in enumerate(agents):
                env.action_space(agent).seed(seed * len(agents) + index)
            if episode == 0:
                for agent in agents:
                    print(f"{agent}: observation={observations[agent].shape}, "
                          f"action={env.action_space(agent)}")
            returns = dict.fromkeys(agents, 0.0)
            steps = 0
            success = False
            while env.agents:
                actions = {agent: env.action_space(agent).sample() for agent in env.agents}
                observations, rewards, terminated, truncated, _ = env.step(actions)
                steps += 1
                for agent, reward in rewards.items():
                    returns[agent] += float(reward)
                success = capture_success(rewards, terminated)
                if success or all(truncated.values()):
                    break
            record = {"episode": episode + 1, "seed": seed, "steps": steps,
                      "success": success, "returns": returns}
            records.append(record)
            print(json.dumps(record, ensure_ascii=False))
    finally:
        env.close()

    summary = {
        "policy": "random actions for all four agents (no training)",
        "mpe2_version": version("mpe2"),
        "episodes": len(records),
        "max_cycles": args.max_cycles,
        "success_rate": sum(item["success"] for item in records) / len(records),
        "mean_steps_including_failures": sum(item["steps"] for item in records) / len(records),
        "records": records,
    }
    print(f"Completed {len(records)} episodes; "
          f"success_rate={summary['success_rate']:.1%}; "
          f"mean_steps={summary['mean_steps_including_failures']:.2f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Results saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
