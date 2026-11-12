# MAPPO 基线架构与接口

## 任务合同

保留现有 SquareTagEnv：3追1逃2障碍，场地半边长1.5，100步有限时域。训练逃逸者固定为边界感知 flee。排除初始追逃接触，最多重采样100次；其他原生重叠保留，重采样延续环境 RNG。

训练适配层统一奖励：`float(success) - 0.01`，即成功步0.99、其他步-0.01。多人同时接触只加一次。成功通过场景接触函数判定，独立于训练奖励。最后一步接触成功优先；后端原生终止/超时字典另存在 info。

## 模块与调用

`Runner -> Policy.act / Policy.value -> PursuitEnv.step -> RolloutBuffer -> GAE -> MAPPO.update`。

训练入口是 scripts/train.py；评估入口 scripts/evaluate.py；演示入口 scripts/render_policy.py。三个入口共享环境适配层，不修改旧的演示任务。

- `reset(seed, opponent_seed) -> Frame(obs,state,info)`
- `step(pursuer_actions[3]) -> Transition(next_frame,reward,success,timeout,info)`
- `Transition.done = success or timeout`
- `policy.act(obs, deterministic=False) -> actions, log_probs`，采样不保留计算图。
- `policy.value(state)` 返回可求导的团队价值；采样时使用 no_grad。
- `policy.evaluate_actions(obs, actions)` 返回可求导的新动作对数概率和熵。

环境不自动重置。Runner 先处理实际转移与结束标志再 reset。回合结束后，缓存下一行可以是新回合起点，但 done 掩码阻止自举和优势跨越边界；实际终态的值不参与学习。仅采集批次结束时对未结束任务使用末状态价值自举。

## 数组约定

Actor：原生16维观测加剩余时间，obs=[3,17]。Critic：state=[21]，位置0:8、速度8:16、障碍位置16:20、时间20:21；顺序追捕者0、1、2、逃逸者。Actor不接收集中式状态。

缓存 obs=[T+1,E,3,17]，state=[T+1,E,21]，actions/old_log_prob=[T,E,3]，奖励/优势/目标=[T,E]，团队价值=[T+1,E]。优势广播到三位 Actor，使用个体概率比，不使用联合动作概率比。Critic每个团队转移只回归一次。

旧概率和价值目标在本批更新中固定，原始优势先构造价值目标，再独立标准化供 Actor 使用。单进程顺序推进多个环境；每批交互数T*E，不乘3。最后一批可缩短，严格达到配置预算，不偷偷超支。

## 评价与模型

开发默认训练种子42。训练重置种子从独立环境流在[0,1000000)抽取；验证种子从1000000开始，每次固定同一组。规则随机源单独设种子。验证保存并恢复 Python/NumPy/Torch 随机数状态，不更新参数，采用最大概率动作。

定期验证约每10000步、20回合，可能因完整更新批次而稍晚；记录实际交互步数。主模型为固定预算末尾 final.pt，不按测试成绩挑选。首次未训练模型也进行验证，作为曲线起点。

模型包含 Actor/Critic、完整配置、交互步数、种子、版本、源代码哈希、上游提交；当前归一化统计为None。加载使用weights_only=True。文件只用于可信项目模型；不提供精确断点续训接口。

跨平台采用 pathlib、标准 argparse、CPU默认与显式可选CUDA，不使用fork或个人绝对路径。当前Mac验证不等于已在Windows实际验证，也不承诺跨设备位级一致。

## 初始配置

两层64单元Tanh；Actor/Critic各3e-4；gamma=.99；GAE lambda=.95；clip=.2；PPO epochs=5；mini-batch数=1；entropy=.01；梯度范数上限=.5。无价值裁剪或价值归一化，Critic为MSE。初始超参数尚未调优。

## 当前边界

只实现原始MAPPO。没有IPPO、协同塑形、RNN、多进程、双方训练或避障留出对手。smoke与5万步短训练仅验证实现与初步学习，不代表正式多种子比较。

## 短训练发现的物理数值修复

硬墙会使两个实体精确重合，上游碰撞法向 delta_pos/dist 在 dist=0 时未定义，实际导致一次短训练验证中断。SquareWorld 现在只对零距离分支用实体顺序决定的反对称对角单位法向；所有非零距离仍直接调用原上游公式。这为原先未定义的碰撞情形补充规则，不是奖励或策略改进。新增有限性、反对称和非零距离一致性测试；原失败记录保留，新短训练从头重跑。规则基线和学习方法均使用同一修复环境。
