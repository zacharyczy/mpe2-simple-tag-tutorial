"""提取自当前项目规则演示；保持硬墙感知 flee 行为。"""
import numpy as np

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


class RulePolicy:
    """Greedy pursuit evaluated through exactly the same task adapter as MAPPO."""
    def __init__(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode
        return self

    def eval(self):
        return self.train(False)

    def act(self, obs, deterministic=True):
        obs = np.asarray(obs)
        directions = obs[..., 12:14].reshape(-1, 2)
        actions = np.array([direction_to_action(x) for x in directions], dtype=np.int64)
        return actions.reshape(obs.shape[:-1]), None
