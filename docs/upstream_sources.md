# MAPPO 上游来源与移植范围

来源：<https://github.com/marlbenchmark/on-policy>

固定提交：`de66d7a4b23fac2513f56f96f73b3f5cb96695ac`。

MIT 原始许可证完整保存在 `third_party/MAPPO_LICENSE`，上游文件 SHA256 见 `third_party/mappo_sources.json`。派生模块保留版权和许可证引用。

| 本项目模块 | 上游来源 | 保留与适配 |
| --- | --- | --- |
| `mappo.py` | `onpolicy/algorithms/r_mappo/r_mappo.py` | 移植概率比、双 surrogate 取最小、熵奖励、分离 Actor/Critic 优化与梯度裁剪；按团队转移组织 mini-batch，对三个 Actor 取平均 |
| `rollout_buffer.py` | `onpolicy/utils/shared_buffer.py` | 移植反向 GAE 与 return=advantage+value；把成功和有限时域上限均作为任务终点；每个环境只存一个团队价值 |
| `policies/actor_critic.py` | `onpolicy/algorithms/r_mappo/algorithm/r_actor_critic.py` 及其 MLP 组件 | 保留 Actor/Critic 分离、分类动作分布、正交初始化；改为两层 Tanh 前馈网络与团队价值输出 |

本版是面向本项目的前馈共享策略 MAPPO 适配基线，不是作者仓库整体复制，也不声称复现论文结果。与上游不同：不使用 RNN、死亡掩码、图像网络、PopArt、价值归一化、特征归一化、价值裁剪或 Huber loss。Critic 使用未乘 1/2 的 MSE；学习率与其他超参数见配置，不隐含沿用作者默认值。

当前无协同创新奖励；团队奖励为统一基础任务定义。硬墙环境与边界感知逃逸规则来自本项目已有代码。原始参考版本从临时目录读取，运行时不依赖临时目录或完整作者仓库。
