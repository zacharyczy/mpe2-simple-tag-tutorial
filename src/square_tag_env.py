"""项目自定义的有限场地 Simple Tag；复用 MPE2，不修改安装目录。

固定 3 追捕者、1 逃逸者、2 障碍物，保持原观测、动作、奖励和出生分布。
创新/环境改动：在物理积分后、奖励与捕获判断前施加正方形硬边界。
"""

import argparse
import math

import numpy as np
import pygame
from mpe2._mpe_utils.simple_env import SimpleEnv, make_env
from mpe2.simple_tag.simple_tag import ExtendedWorld, Scenario
from pettingzoo.utils.conversions import parallel_wrapper_fn

DEFAULT_ARENA_HALF_SIZE = 1.5


def arena_half_size(value):
    """最小 1.1，容纳原障碍物出生范围 ±0.9 及其 0.2 半径。"""
    try:
        size = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("arena half size must be a number") from exc
    if not math.isfinite(size) or size < 1.1:
        raise argparse.ArgumentTypeError("arena half size must be finite and >= 1.1")
    return size


class SquareWorld(ExtendedWorld):
    def __init__(self, half_size):
        super().__init__()
        self.half_size = arena_half_size(half_size)

    def get_collision_force(self, entity_a, entity_b):
        """定义完全重合时的接触法向，避免上游 delta_pos / 0 产生 NaN。

        非零距离完全复用上游；零距离用实体顺序决定对角单位法向，
        交换实体时法向反号。此处不添加随机数或修改实体实际位置。
        """
        if (entity_a is entity_b or not entity_a.collide or not entity_b.collide
                or np.linalg.norm(entity_a.state.p_pos - entity_b.state.p_pos) > 0):
            return super().get_collision_force(entity_a, entity_b)
        sign = 1.0 if self.entities.index(entity_a) < self.entities.index(entity_b) else -1.0
        normal = np.full(self.dim_p, sign / np.sqrt(self.dim_p))
        distance_min = entity_a.size + entity_b.size
        penetration = np.logaddexp(0, distance_min / self.contact_margin) * self.contact_margin
        force = self.contact_force * normal * penetration
        return [force if entity_a.movable else None,
                -force if entity_b.movable else None]

    def integrate_state(self, p_force):
        # 上游仍负责阻尼、控制、最高速度和实体碰撞。
        super().integrate_state(p_force)
        for entity in self.entities:
            if not entity.movable:
                continue
            limit = self.half_size - entity.size
            entity.state.p_pos = np.clip(entity.state.p_pos, -limit, limit)
            # 非弹性墙：去掉朝外的法向速度，保留切向与朝内速度。
            outward = ((entity.state.p_pos >= limit) & (entity.state.p_vel > 0)) | (
                (entity.state.p_pos <= -limit) & (entity.state.p_vel < 0)
            )
            entity.state.p_vel[outward] = 0.0


class SquareTagEnv(SimpleEnv):
    metadata = {**SimpleEnv.metadata, "name": "square_tag_v0"}

    def __init__(self, arena_half_size=DEFAULT_ARENA_HALF_SIZE, max_cycles=100,
                 render_mode=None):
        scenario = Scenario(terminate_on_success=True)
        upstream_world = scenario.make_world(num_good=1, num_adversaries=3, num_obstacles=2)
        world = SquareWorld(arena_half_size)
        # 复用上游实体和场景，不复制场景或物理引擎实现。
        world.agents = upstream_world.agents
        world.landmarks = upstream_world.landmarks
        world.dim_c = upstream_world.dim_c
        super().__init__(scenario=scenario, world=world, max_cycles=max_cycles,
                         render_mode=render_mode, continuous_actions=False,
                         dynamic_rescaling=False)

    def draw(self):
        """固定相机并画出实际墙；实体半径和坐标使用同一比例。"""
        self.screen.fill((250, 250, 250))
        side = min(self.width, self.height) - 100
        scale = side / (2 * self.world.half_size)
        rect = pygame.Rect((self.width - side) // 2, (self.height - side) // 2, side, side)
        pygame.draw.rect(self.screen, (255, 255, 255), rect)
        pygame.draw.rect(self.screen, (65, 65, 65), rect, 3)
        for entity in self.world.entities:
            x, y = entity.state.p_pos
            center = (round(self.width / 2 + x * scale), round(self.height / 2 - y * scale))
            radius = max(1, round(entity.size * scale))
            pygame.draw.circle(self.screen, entity.color * 200, center, radius)
            pygame.draw.circle(self.screen, (0, 0, 0), center, radius, 1)
        label = f"Square arena: [-{self.world.half_size:g}, {self.world.half_size:g}] | step {self.steps}"
        self.game_font.render_to(self.screen, (50, 20), label, (40, 40, 40), size=18)


parallel_env = parallel_wrapper_fn(make_env(SquareTagEnv))
