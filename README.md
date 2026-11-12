# MAPPO 学习基线

新增前馈共享策略 MAPPO；原来的规则演示与环境导读保留在下方。新增训练适配使用成功奖励1、每步代价0.01，与旧演示的原生接触奖励10不同。

详细接口见 [MAPPO 架构](docs/mappo_architecture.md)，代码来源见 [上游移植说明](docs/upstream_sources.md)。

本次验证见 [实施与验证记录](docs/implementation_validation.md)：23项测试通过，完成5万步单种子短训练；最终20回合开发验证成功率为0%，尚未显示有效学习。

## 安装与运行

使用自己选定的 Python 环境；项目不自动创建或激活虚拟环境。若需要隔离环境，可自行创建并激活后执行。当前 MAPPO 开发使用 Python 3.12；安装器会检查各平台依赖可用性。

```bash
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m unittest discover -s tests -v
python scripts/train.py --config configs/mappo_smoke.json --output outputs/mappo_smoke
python scripts/train.py --config configs/mappo_baseline.json --output outputs/mappo_seed42
```

若当前环境没有构建工具，先执行 `python -m pip install setuptools`。Windows PowerShell 也可使用上述命令。输出目录必须为空，防止覆盖既有实验。通过 `--seed 43` 等可以分别训练独立模型，每次使用不同输出目录；这不是精确断点续训。

加载模型，用开发/验证条件检查（不是留出测试）：

```bash
python scripts/evaluate.py --checkpoint outputs/mappo_seed42/final.pt --episodes 20 --seed-start 1000000 --output outputs/mappo_seed42/loaded_validation.json
python scripts/evaluate.py --rule chase --config configs/mappo_baseline.json --episodes 20 --seed-start 1000000 --output outputs/rule_validation.json
python scripts/render_policy.py --checkpoint outputs/mappo_seed42/final.pt --gif outputs/mappo_seed42/demo.gif
```

不指定 `--gif` 时打开人类可见窗口。`evaluate.py` 默认200回合，并支持 `--opponent random`；正式测试必须使用预先留出的种子，不能将已用于开发的种子再次声称为独立测试。

## 输出内容

- `config.json`、`provenance.json`：完整配置、代码与依赖版本、未提交源码哈希。
- `metrics.jsonl`、`validation_*.json`：验证曲线与回合级原始记录。
- `episodes.jsonl`：训练回合的成功、步数与累计团队奖励。
- `checkpoint_*.pt`、`final.pt`：定期快照与固定预算末尾模型。
- `validation_curve.png`：成功率和包含失败的平均步数曲线。

所有模型、日志、曲线与演示在 outputs 内，默认不提交Git。requirements-lock.txt记录Mac开发环境实际版本，不保证所有平台的CUDA/二进制轮子相同。Windows MAPPO运行仍需在目标机器实测。

---

# Simple Tag 入门与追逃决策代码导读

## 克隆后快速运行

需要 Python 3.11 或以上版本，当前已验证 Windows 和 Python 3.13.12。使用你选定的 Python 环境，在终端执行：

```powershell
git clone https://github.com/zacharyczy/mpe2-simple-tag-tutorial.git
cd mpe2-simple-tag-tutorial
python -m pip install -r requirements.txt
python src/simple_tag_demo.py --episodes 2
python src/simple_tag_policy_demo.py --pursuers chase --evader flee --episodes 2
```

后续命令均在仓库根目录执行。已有依赖时可以直接运行；默认不会创建虚拟环境。需要无窗口运行时加 `--render none`。协作流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。

想先了解库与动画窗口的整体关系，可先读 [MPE2 库与演示动画说明](MPE2-library-and-demo-guide.md)，再按本说明逐步阅读和修改代码。

以下两份原有示例提供环境运行和策略接入演示：双方可以随机行动，也可以让追捕者朝逃逸者移动、让逃逸者远离最近追捕者。这两个演示脚本本身不训练神经网络；新增 MAPPO 训练入口见本文开头。当前没有实现 IPPO、协同包抄或避障规划。

这份说明带你回答三个问题：**环境替我们做什么？自己的决策写在哪里？训练算法以后接在哪里？**

说明依据本机 **MPE2 1.1.1** 的实际源码编写。观测切片与教学规则固定针对 **3 个追捕者、1 个逃逸者、2 个障碍物、离散动作、全实体可见** 的配置。改变实体数量、观测设置或版本后，要重新核对。

## 新增正方形场地与边界感知逃离

规则演示 `src/simple_tag_policy_demo.py` 默认使用 **边长 3、范围 $[-1.5,1.5]^2$** 的场地，墙在画面中固定显示。场地比原来的中央出生区域更大，给逃逸者留出转向与沿墙移动空间；这个尺寸是试运行起点，尚未作为正式实验难度定稿。

```powershell
# 默认有限场地，观察边框和沿墙逃离
python src/simple_tag_policy_demo.py --pursuers chase --evader flee --episodes 3
# 放大到边长 4；参数表示半边长，不是完整边长
python src/simple_tag_policy_demo.py --pursuers chase --evader flee --arena-half-size 2 --episodes 3
# 原无墙环境和原 flee 规则，便于对照
python src/simple_tag_policy_demo.py --pursuers chase --evader flee --unbounded --episodes 3
# 无窗口检查
python src/simple_tag_policy_demo.py --pursuers chase --evader flee --render none --episodes 10 --seed 42
# 边界、规则、状态一致性与 Parallel API 自检，无须额外安装测试库
python -m unittest discover -s tests -v
```

环境改动放在 `src/square_tag_env.py`，没有修改 `site-packages`。原 `src/simple_tag_demo.py` 继续使用原始无墙环境，保留为环境检查参考。下面介绍上游初始化与自动缩放的旧示例时，均指原始无墙环境；规则演示的实际创建入口现为 `square_tag_env.parallel_env()`。

### 墙的物理含义与执行时序

- 双方均受墙约束。半径为 $r$ 的智能体中心不能超出 $[-L+r,L-r]^2$，因此整个圆都在墙内。
- 在物理积分后立即校正位置，清除朝墙外的速度，保留沿墙或朝内的速度；不进行弹性反弹。
- 墙的校正在奖励、捕获、终止和下一观测计算之前完成，不只是修改画面。
- 动作仍是 `Discrete(5)`，观测仍是 16/14 维。固定墙的位置由公共场地配置和自身绝对位置确定，不新增墙实体或观测槽位。
- 保留上游速度、奖励公式、首次接触终止和出生分布：智能体中心在 $[-1,1]^2$，障碍物中心在 $[-0.9,0.9]^2$。原初始化可能存在实体重叠，本次没有改为排斥采样。
- 原有逃逸者软越界惩罚也保留，它与硬墙是两回事；规则逃逸者不会根据该惩罚学习。比较场地大小时须记录这个事实。
- `--arena-half-size` 至少为 1.1，以容纳原障碍物出生范围及其半径。增大场地不会自动扩大出生范围。

### flee 怎样选“次优方向”

`select_action()` 的 `flee` 分支调用 `flee_action()`：先找最近的**追捕者**，把四个移动方向按它们在“远离该追捕者”方向上的投影打分，再排除朝近墙外运动的方向，选择剩余最高分动作。

预警区考虑逃逸者半径 0.05、额外余量 0.08 和当前向墙速度未来两个物理步长的位移。它使用自身观测中的位置与速度，以及公共场地配置，不读取环境内部世界状态。处在右墙附近时，向右不可用；若威胁来自左下，则选择向上。角落里若两个向外方向都不可用，就选剩余方向中较好的一个，即使这个动作暂时会接近威胁。投影同分时按动作编号打破平局（左右先于上下，上下同分时向下优先）。

这仍是单步教学规则，没有障碍物避让、惯性精确预测或协作规划。保留约 0.08 的墙边余量也不代表永不碰墙：惯性和实体碰撞仍可能把它推到墙上，硬边界负责兜底。`--unbounded` 同时恢复无墙环境和旧 flee，不能把两种模式的结果差异全部归因于单一改动；需要消融时，应固定同一场地单独比较规则。

正式比较 IPPO、MAPPO 和改进方法时，要共用同一场地、逃逸规则、出生种子、速度和回合上限，不能混用原随机示例的无墙成绩。

### 本次试运行记录

2026-11-12，在 Python 3.12.2、MPE2 1.1.1、PettingZoo 1.27.0 上通过 11 项自动检查（包括 Parallel API、1000 步随机交互的边界不变量、角落择向、渲染与奖励/观测的一致性）。使用 `chase` 对 `flee`、种子 42–71、每回合最多 100 步试运行：

| 环境与规则 | 捕获回合数 | 包含失败的平均步数 |
| --- | ---: | ---: |
| 边长 3，边界感知 flee（默认） | 20/30 | 72.67 |
| 边长 4，边界感知 flee | 13/30 | 78.07 |
| 原无墙环境，旧 flee | 1/30 | 96.90 |

默认尺寸暂取边长 3：这批初始条件下既有捕获也有逃逸，适合观察转向与沿墙运动。它们是规则演示的试运行数据，没有训练神经网络，也不证明某种场地或规则在一般情形更优。可用上方命令加 `--episodes 30 --max-cycles 100 --seed 42` 复现。


## 先定位你真正要看的代码

### Python 文件是什么关系

**两个 demo 文件都是本项目的调用示例，不是 MPE2 库的原版源码。** `simple_tag_demo.py` 是我最先写的随机动作示例；`simple_tag_policy_demo.py` 是后来写的、已经接上简单追踪和逃离规则的教学示例。

两个文件都能单独运行。教学示例通过 `from simple_tag_demo import capture_success, positive_int` 只复用两个辅助函数；导入时不会运行随机演示的 `main()`，不会同时启动另一个环境。这是文件末尾 `if __name__ == "__main__"` 的作用。

**现在要实践自己的决策，直接改 `src/simple_tag_policy_demo.py` 即可，不需要再新建文件，也不用两个文件一起运行。** 原随机示例可以保留用于对照。以后策略很多时，才有必要另建模块再 `import`；那只是组织代码，不是库要求的接入步骤。

### 智能体在哪里设置

打开 `src/simple_tag_policy_demo.py`，在 `main()` 中找到这段：

```python
from square_tag_env import parallel_env

env = parallel_env(arena_half_size=args.arena_half_size, max_cycles=args.max_cycles,
                   render_mode="human")
```

这一调用通过项目场地封装，复用库创建 1 个逃逸者、3 个追捕者、2 个障碍物，不需要你再写 `Agent()`。创建后，`env.agents` 中的名称是 `adversary_0`、`adversary_1`、`adversary_2`、`agent_0`。循环中的 `agent` 是其中一个名称字符串，不是你新定义的智能体类。

如果想看真正的智能体对象在哪里创建，打开库文件 `<Python安装目录>\Lib\site-packages\mpe2\simple_tag\simple_tag.py`，搜索 `def make_world` 和 `world.agents =`。当前只是学习和控制动作时，不必改这个库文件。

### Observation 是谁设置 谁产生的

**默认向量包含哪些内容，由库中的 `Scenario.observation()` 定义；每一步的具体数值，由环境根据当时的世界状态自动生成。** 你不必在策略代码中手工创建 16 个或 14 个观测值。

规则演示的场地环境固定使用全实体观测；原始环境的 `num_agent_neighbors=None, num_landmark_neighbors=None` 控制可见实体范围；这是配置，不是在这里逐个定义观测字段。字段排列在库文件 `simple_tag.py` 的 `def observation` 中，详细切片见本说明第 5 节。

在项目里，观测从这两处进入你的程序：

```python
observations, infos = env.reset(seed=seed)  # 开始时的观测
observations, rewards, terminated, truncated, infos = env.step(actions)  # 下一时刻观测
```

`observations` 是全体智能体的字典，`observations["adversary_0"]` 才是其中一个向量。调用 `select_action(..., current_observations[agent], ...)` 后，该向量在函数内部的参数名变成单数 `observation`，两者没有不同的数据来源。

`env.observation_space("adversary_0")` 描述向量的形状、类型和范围；它不是当前的具体观测值。不要把 observation space 和 observation 数组混淆。

### Discrete 在哪里

正确拼写是 `Discrete`。项目代码里设置 `continuous_actions=False`，库内部因此创建 `Discrete(5)`，你不用重复写 `Discrete(5)`。

创建的位置是 `<Python安装目录>\Lib\site-packages\mpe2\_mpe_utils\simple_env.py`，搜索 `spaces.Discrete(space_dim)`。当前 `space_dim` 是 5，所以可以选择动作 0、1、2、3、4。

要在自己的程序里查看它，可在 `reset()` 后插入：

```python
print(env.agents)                                 # 四个名称
print(env.action_space("adversary_0"))            # Discrete(5)：动作规则
print(env.observation_space("adversary_0"))       # Box(...)：观测规则
print(observations["adversary_0"])                # 当前实际的 16 个数
```

### 我到底改哪几行 怎么才算接上

在教学示例的 `select_action()` 中搜索 `if is_pursuer and policy == "chase":`。这个分支就是你写追捕决策的地方；下一段 `if not is_pursuer and policy == "flee":` 是写逃逸决策的地方。函数每次返回一个动作整数。

先做一个明确的小实验：暂时将追捕者 `chase` 分支改成下面这样，其他分支保持原样：

```python
if is_pursuer and policy == "chase":
    return 2  # 先让三个追捕者每步都选择向右；只用于核对接入
```

保存后运行（先确保终端位于项目根目录）：

```powershell
python '.\src\simple_tag_policy_demo.py' --pursuers chase --evader flee --render none --episodes 1 --max-cycles 1 --inspect
```

打印的 `actions_t` 中三个 `adversary_*` 应全为 2。**这就说明你的函数返回值已经经过动作字典传入环境，不需要任何额外注册。** 注意一定带上 `--pursuers chase`，否则默认走 `random` 分支，你修改的代码就不会执行。

核对后，恢复为 `return direction_to_action(observation[12:14])`，再逐步写自己的规则。`flee` 分支同理，通过 `--evader flee` 启用。动作选好后，`main()` 已经负责调用 `select_action()`、收集四个返回值和执行 `env.step(actions)`，这些接线不必重写。

### 观测只能在运行结束后的终端看到吗

不是。它在 `reset()` 或每次 `step()` 返回时，就已经是内存中的数组，策略也是当场读取它，而不是从终端输出中读取。

- 终端查看：`--inspect` 在运行过程中打印首回合第一步的观测与动作，不必等到全部运行结束。原随机脚本默认只打印观测的 `.shape`，不是完整数组。
- 调试查看：在 VSCode 中于教学示例的 `current_observations = observations` 那行设置断点，调试启动该文件，暂停时在“变量”或“监视”中查看 `observations["adversary_0"]`。只点普通运行按钮不会停在断点。
- 持续观察：临时在 `while env.agents:` 中、生成动作前加 `print(observations["adversary_0"])`，每步都会打印；看完移除，避免日志过多。
- 保存查看：可以把各数组用 `.tolist()` 转成列表后存入 JSON。当前两个脚本没有保存完整观测历史，原脚本的 `--output` 只保存回合统计。

图形窗口默认只画实体，不会把完整观测向量作为文字显示。想在画面上叠加数值，需要另外实现显示功能。

## 建议的阅读路线

1. 先读第 1–3 节，理解环境与策略的分工，运行原随机演示。
2. 打开 `src/simple_tag_demo.py`，跟随第 4 节从 `main()` 读到交互循环。
3. 读第 5–7 节，弄清观测、动作、奖励和结束信号。
4. 运行第 8 节，打印并核对一次交互。
5. 打开 `src/simple_tag_policy_demo.py`，跟随第 9 节理解双方决策接入。
6. 按第 10–11 节切换策略，再自己修改一条规则。
7. 读第 12 节，区分规则、模型推理和模型训练；最后按需查阅第 13–16 节。

第一遍不必读懂物理引擎，也不必先学习完整的强化学习算法。

## 1 环境与决策分别由谁负责

### 1.1 三个名称的关系

**MPE2** 是一组多智能体仿真环境。**Simple Tag** 是其中的追捕逃逸场景，通过 `from mpe2 import simple_tag_v3` 导入。`v3` 是场景接口的版本标识，不是强化学习算法。

**PettingZoo** 提供多智能体交互接口规范。本项目使用 Parallel API：将所有当前智能体的动作放进一个字典，一次提交。

**策略（policy）** 是我们提供的决策方法：读取信息，输出动作。随机采样、手写规则和训练过的神经网络都可以充当策略。环境只执行动作，不会自动替你学习。

### 1.2 Simple Tag 已经提供什么

- 创建智能体和障碍物，定义名称、大小、颜色和运动参数。
- 随机初始化位置，重置回合。
- 根据动作计算运动、速度、阻尼和碰撞响应。
- 为各智能体生成观测，并定义合法动作空间。
- 计算每步奖励、任务终止和超时信号。
- 显示图形窗口，也支持返回画面像素的 `rgb_array` 模式。本项目的命令行暂未开放后者。

### 1.3 你需要提供什么

- 三个追捕者每一步分别选什么动作。
- 逃逸者每一步选什么动作。即使它只执行规则，也要给它动作。
- 多回合运行、数据记录和实验评价。
- 如需学习：网络、经验收集、训练目标、参数更新和模型保存。
- 如需改任务：自己的奖励、观测限制或终止约定，并保证比较条件一致。

课程基础任务“只训练追捕方”是指：**仍然为双方提供动作，但只更新追捕方模型参数；逃逸方使用预先定义的规则。**

## 2 项目文件各做什么

- [`src/simple_tag_demo.py`](src/simple_tag_demo.py)：原随机演示。创建环境、随机采样、推进仿真、累计奖励、打印和保存结果。先读它。
- [`src/simple_tag_policy_demo.py`](src/simple_tag_policy_demo.py)：本次新增的教学示例。把决策提取成函数，可分别选择双方的策略，并打印首步数据。
- [`requirements.txt`](requirements.txt)：固定需要安装的 MPE2 版本。
- [`requirements-lock.txt`](requirements-lock.txt)：记录本项目及其依赖的实际版本，不包含其他项目的全部包。
- [`setup.ps1`](setup.ps1)：使用现有 Python 补装依赖并检查依赖关系，不创建虚拟环境。
- [CONTRIBUTING.md](CONTRIBUTING.md)：分工、分支、提交、Pull Request 和实验配置约定；个人编辑器配置不提交。
- `outputs/`：原随机演示保存的 JSON 结果。两个演示都不会生成训练模型。

项目代码在 `src/`；Simple Tag 库源码在 `<Python安装目录>\Lib\site-packages\mpe2`。接入策略先改项目文件，不要直接改 `site-packages`，以免影响其他项目或在重装时丢失改动。

## 3 先运行原随机演示

在 PowerShell 中执行：

```powershell
# 请先进入你克隆得到的仓库根目录
python '.\src\simple_tag_demo.py' --episodes 2 --max-cycles 100
```

三个红色圆是追捕者，一个绿色圆是逃逸者，两个大黑圆是静态障碍物。程序自动更新画面，结束一个回合后重置，全部回合结束后关闭窗口。

此时双方都随机行动。偶尔接触成功，只能说明发生了接触，不能说明学会了合作。

无窗口运行并保存结果：

```powershell
python '.\src\simple_tag_demo.py' --render none --episodes 10 --seed 42 --output '.\outputs\random_smoke_test.json'
```

参数含义：

- `--episodes 10`：运行 10 个回合，每个回合都会重置。
- `--max-cycles 100`：每回合最多推进 100 次世界状态。一次包含四个智能体各一个动作。
- `--seed 42`：首回合种子 42，后续使用 43、44 等；环境和随机动作空间分别设种子。
- `--render none`：不显示窗口，通常运行更快。默认是 `human`。
- `--output ...`：原随机演示支持写入 JSON；不指定时只打印。

输出中的 `steps` 是整个世界推进的次数，不是四个智能体动作数之和。`returns` 是各智能体本回合的累计奖励。平均步数包含失败回合，失败按上限计入；成功率是成功回合数除以总回合数。

## 4 从 main 开始逐步阅读原代码

打开 `src/simple_tag_demo.py`，用 VSCode 搜索下列函数名或代码片段，按数据流阅读。

### 第一步 程序从哪里开始

```python
if __name__ == "__main__":
    main()
```

直接运行文件时调用 `main()`。上面定义的函数不会自动开始仿真。`positive_int()` 只检查命令行参数是否为正整数，不参与决策，可以先略读。

### 第二步 参数如何传入环境

找到 `parser.add_argument(...)` 和 `args = parser.parse_args()`。例如 `--max-cycles 100` 通过 `args.max_cycles` 传给环境。

命令中的 `--render none` 会被转换成 Python 的 `None`，而不是给环境传入字符串 `"none"`：

```python
render_mode="human" if args.render == "human" else None
```

### 第三步 创建场景

```python
env = simple_tag_v3.parallel_env(
    num_good=1,
    num_adversaries=3,
    num_obstacles=2,
    max_cycles=args.max_cycles,
    continuous_actions=False,
    terminate_on_success=True,
    render_mode="human" if args.render == "human" else None,
)
```

`num_adversaries` 是追捕者数量，`num_good` 是逃逸者数量。名字来自场景，不表示学习对象一定是 good 一方。

`continuous_actions=False` 选择五个离散动作。`terminate_on_success=True` 启用场景成功终止。当前只有一个逃逸者，任意追捕者接触它即可结束；若增加多个逃逸者，这个选项要求所有逃逸者同时被接触，不再等价于“任意一人被抓就结束”。

库默认是 25 步、不开启成功终止，本项目显式使用 100 步和接触结束。未启用 `curriculum` 难度递增，也没有限制最近可见实体数量。

### 第四步 开始一个新回合

```python
observations, _ = env.reset(seed=seed)
agents = list(env.possible_agents)
```

`reset()` 重置位置、速度、计数等，返回 `observations` 和 `infos` 两个字典。`_` 只是普通变量名，表示暂时忽略该返回值。

观测不是图像，而是按智能体名称索引的数值向量：

```python
observations["adversary_0"]  # 第一个追捕者的观测，16 个数
observations["agent_0"]      # 唯一逃逸者的观测，14 个数
```

`env.possible_agents` 是该环境所有可能出现的智能体；`env.agents` 是当前仍参与交互的智能体。初始化统计用前者，生成当前动作使用后者。

随后 `env.action_space(agent).seed(...)` 为动作采样器设种子。只给 `reset()` 设种子，不会自动控制你自己代码中的全部随机源。

### 第五步 找到决策接入位置

```python
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
```

这行字典推导式相当于：

```python
actions = {}
for agent in env.agents:
    actions[agent] = env.action_space(agent).sample()
```

它为每个智能体随机抽动作，没有读取 `observations[agent]`，所以不知道目标、队友或障碍物在哪里。

一个合法动作字典可能是下面这样，数字只是示例，不是固定运行结果：

```python
actions = {
    "adversary_0": 2,
    "adversary_1": 4,
    "adversary_2": 1,
    "agent_0": 3,
}
```

**接入自己的决策，就是将 `.sample()` 换成“读当前观测、返回动作”的函数调用。双方都在这里接入，不需要向 MPE2 注册策略，也不需要改 `env.step()`。**

### 第六步 一起提交双方动作

```python
observations, rewards, terminated, truncated, _ = env.step(actions)
```

环境执行这个动作组合，推进一次世界状态，返回五个按名称索引的字典：

- `observations`：下一时刻的观测。
- `rewards`：刚执行这一步得到的奖励，不是整回合累计奖励。
- `terminated`：是否因任务终止条件结束。
- `truncated`：是否因步数上限等外部限制截断。
- `infos`：补充信息。当前默认配置通常是每个名称对应空字典，不会自动包含成功率或模型信息。

左侧重新赋值会覆盖旧的 `observations`。下一轮基于新观测决策；以后训练时要先保存旧观测，见第 12 节。

循环先后计算四个动作，不代表世界里四个智能体轮流移动。只有调用 `step(actions)` 才推进世界，它们都基于同一时刻的观测决策。

这里的 Parallel 描述的是同时提交动作的交互语义，不要求你编写多线程或使用 GPU。

### 第七步 统计并判断是否结束

```python
for agent, reward in rewards.items():
    returns[agent] += float(reward)
success = capture_success(rewards, terminated)
if success or all(truncated.values()):
    break
```

累计奖励只是在统计，不会更新策略。`capture_success()` 是本项目辅助函数，不是库 API，限制见第 7 节。

`break` 结束本回合的 `while`；外层 `for episode ...` 开始下一回合。`finally: env.close()` 负责释放窗口等资源，最后生成汇总并按需保存。

## 5 观测向量中的每一段是什么

本节切片仅适用于当前固定配置。Python 切片左端包含、右端不包含：`[12:14]` 取索引 12、13 两个数。

### 5.1 追捕者的 16 维观测

对任意一个 `adversary_i`：

```python
obs[0:2]    # 自身速度 vx, vy
obs[2:4]    # 自身绝对位置 x, y
obs[4:6]    # 障碍物 0 相对于自身的位置 dx, dy
obs[6:8]    # 障碍物 1 相对于自身的位置 dx, dy
obs[8:10]   # 另一个追捕者相对于自身的位置
obs[10:12]  # 另一个追捕者相对于自身的位置
obs[12:14]  # 唯一逃逸者相对于自身的位置
obs[14:16]  # 逃逸者速度 vx, vy，是速度本身而非相对速度
```

为什么三个追捕者都能用 `[12:14]` 找逃逸者？实体顺序是 `adversary_0, adversary_1, adversary_2, agent_0`，生成观测时去掉自身，逃逸者仍位于其他智能体位置的最后一段。

但两个队友的顺序随观察者变化：追捕者 0 看到队友 1、2；追捕者 1 看到队友 0、2；追捕者 2 看到队友 0、1。因此 `[8:10]` 不永远对应某个固定名称。

### 5.2 逃逸者的 14 维观测

对于 `agent_0`：

```python
obs[0:2]    # 自身速度 vx, vy
obs[2:4]    # 自身绝对位置 x, y
obs[4:6]    # 障碍物 0 相对于自身的位置
obs[6:8]    # 障碍物 1 相对于自身的位置
obs[8:10]   # adversary_0 相对于自身的位置
obs[10:12]  # adversary_1 相对于自身的位置
obs[12:14]  # adversary_2 相对于自身的位置
```

当前实现只附加“其他逃逸者”的速度，不附加追捕者速度。唯一逃逸者没有其他逃逸队友，所以没有额外速度段。

“全实体可见”指所有其他实体的位置都提供，**不等于包含所有其他实体的速度**。这解释了 16 维与 14 维的区别。

### 5.3 相对位置的方向

相对位置等于“其他实体位置减去自身位置”。若自己在 `[0.2, -0.1]`、目标在 `[0.8, 0.3]`，相对位置就是 `[0.6, 0.4]`，表示目标在右上方。

可以用 `obs[2:4] + obs[12:14]` 恢复追捕者看到的逃逸者绝对位置，但最简单的追踪规则直接用相对位置就足够。

### 5.4 为什么不能随便沿用切片

增加障碍物会挪动后面的索引，增加逃逸者会改变位置和速度段。开启 `num_agent_neighbors` 后，还会按最近实体选择并填充槽位。

教学示例检查了向量形状，但相同形状也不能证明语义相同，例如最近邻排序可能已变化。因此它显式固定配置，不作为通用观测解析器。

## 6 动作到底控制什么

当前动作空间是 `Discrete(5)`，应返回整数：

- `0`：不施加主动运动控制。
- `1`：向左，负 x。
- `2`：向右，正 x。
- `3`：向下，负 y。
- `4`：向上，正 y。

不能直接传 `[dx, dy]`、任意坐标或动作字符串。环境不会自动把目标方向变成路径，你要自己选择合法动作。

动作控制运动作用方向，不是瞬移，也不是直接设速度。环境中有惯性、阻尼、碰撞响应，**动作 0 不会立即将速度清零**。

本机版本先用当前速度更新位置，再更新速度。静止状态下首次向右可能先表现为速度变化，下一步位置才明显变化。单步检查时不要因此认为动作没有执行。

当前默认参数：追捕者最高速度 1.0、控制加速度参数 3.0、半径 0.075；逃逸者分别为 1.3、4.0、0.05；障碍物半径为 0.2。逃逸者更快。中心距离严格小于两者半径之和时，场景判定接触；这些是仿真单位，不是像素。

若以后设 `continuous_actions=True`，动作会变为范围 `[0, 1]`、形状 `(5,)` 的向量，需要适配策略输出，不能继续直接传动作整数。

## 7 奖励 成功与回合结束

### 7.1 默认奖励的实际含义

本机 `Scenario.adversary_reward()` 使用团队接触奖励：**每出现一对追捕者与逃逸者接触，三个追捕者各得到 +10。** 不仅是实际碰到目标的那一名追捕者得分。

例如只有追捕者 2 接触，三个追捕者本步奖励通常都是 10；同一步有两个追捕者接触，三个追捕者的本步奖励都可以是 20。

逃逸者每与一个追捕者接触得 -10，另外对自身 x、y 坐标分别计算靠近或超出区域的惩罚：绝对值小于 0.9 时该坐标无惩罚，从 0.9 开始惩罚递增。

这是奖励惩罚，不是不可越过的墙。原无墙环境的图形相机会自动缩放，窗口边缘不等于惩罚边界；新增有限场地则使用固定相机和可见硬墙。当前未启用距离塑形奖励，默认也没有独立的“撞障碍物扣分”项，障碍物主要提供物理碰撞响应。

### 7.2 奖励不等于成功率

逃逸者即使没被抓，也可能因长时间跑出惩罚区域产生很大的负累计奖励。三个追捕者分数相同，也不能证明它们学出了分工。

课程主要任务指标仍是成功率和包含失败回合的平均步数，协作改进要通过对照实验说明。

### 7.3 terminated 与 truncated 的区别

当前单逃逸者配置中，接触成功通常使所有 `terminated=True`，到达步数上限通常使所有 `truncated=True`。

本机 1.1.1 先检查上限，再检查成功。所以最后一步恰好接触，可能只有超时标志。项目的 `capture_success()` 同时检查终止与追捕方本步正奖励，避免漏计该情况。

该辅助判断只适用于当前单逃逸者、默认奖励和接触终止约定。若加入“靠近目标也给正奖励”等塑形，或改成多逃逸者，必须重新定义成功，不能继续将任意正奖励视为任务成功。

初始位置是独立随机生成的，不保证实体之间留出足够间距，少量回合可能很快接触。正式实验应统一初始条件和是否允许初始重叠的规则。

## 8 打印并核对一次交互

教学示例的 `--inspect` 打印首回合第一步的观测、动作、奖励、结束标志和下一观测：

```powershell
python '.\src\simple_tag_policy_demo.py' --pursuers chase --evader flee --render none --episodes 1 --max-cycles 1 --seed 42 --inspect
```

按顺序核对：

1. 找 `observation_t[adversary_0]`，按第 5 节拆解。
2. 读 `[12:14]`，判断逃逸者相对方向。
3. 看 `actions_t` 中这个追捕者选了什么动作。
4. 看 `rewards_t`，它是执行双方动作后的本步奖励。
5. 比较 `observation_t_plus_1[adversary_0]` 中的位置与速度变化。
6. 本命令上限是 1，预期会超时；它用于观察数据流，不用于评价策略。

种子 42 下，追捕者 0 看到的逃逸者相对位置约为 `[-0.0256, 0.6944]`。纵向差更大且为正，教学追踪规则应选 `4` 向上。你可以核对输出，不必先理解整个物理引擎。

## 9 按三个函数阅读双方决策示例

打开 `src/simple_tag_policy_demo.py`。它复用原文件的 `positive_int()`、`capture_success()`，没有修改原随机演示或库。

### 9.1 direction_to_action 把期望方向变成动作

```python
def direction_to_action(direction):
    dx, dy = direction
    if max(abs(dx), abs(dy)) < 1e-8:
        return 0
    if abs(dx) >= abs(dy):
        return 2 if dx > 0 else 1
    return 4 if dy > 0 else 3
```

一次只能选择一个轴方向，所以优先沿差距更大的轴移动。相等时优先横向是示例自己的约定。

手算：`[0.6, 0.4]` 选 2 向右；`[-0.2, 0.8]` 选 4 向上；`[0.1, -0.5]` 选 3 向下。

### 9.2 select_action 是统一决策入口

```python
select_action(agent, observation, action_space, pursuer_policy, evader_policy)
```

它接受名称、该智能体自己的当前观测、动作空间和双方所选策略名，返回动作整数。用 `agent.startswith("adversary_")` 区分追捕与逃逸身份。

追捕方的 `chase`：

```python
prey_relative_position = observation[12:14]
return direction_to_action(prey_relative_position)
```

相对位置从自己指向目标，直接朝它移动。三个追捕者各执行相同规则，没有包抄、角色分工或路径规划。

逃逸方的 `flee` 先用下面的逻辑计算远离方向（`--unbounded` 下直接返回这个动作）：

```python
pursuer_relative_positions = observation[8:14].reshape(3, 2)
distances = np.linalg.norm(pursuer_relative_positions, axis=1)
nearest = pursuer_relative_positions[int(np.argmin(distances))]
return direction_to_action(-nearest)
```

默认有限场地实际由 `return flee_action(observation, half_size)` 调用完整规则；辅助函数还会排除朝墙外的方向，详见前面的新增场地说明。

6 个数排成三行，每行是一个追捕者相对位置。`norm(..., axis=1)` 得到三个距离；`argmin` 选择最近者；对指向威胁的向量取负号，得到远离威胁的方向。

`random` 分支调用动作空间采样；`stay` 分支固定返回 0，只表示不施加主动控制，不承诺位置固定。

这些规则由本项目提供，不是环境自动安排的内置对手策略。默认有限场地的 `flee` 已加入边界感知与基于速度的预警，但没有避障或长期规划，仍可能逃入障碍物或惩罚区域。`--unbounded` 恢复没有边界感知的旧规则。

### 9.3 main 提交双方动作

```python
current_observations = observations
actions = {
    agent: select_action(
        agent, current_observations[agent], env.action_space(agent),
        args.pursuers, args.evader, half_size,
    )
    for agent in env.agents
}
observations, rewards, terminated, truncated, infos = env.step(actions)
```

与原文件相比，核心是用 `select_action(...)` 替换 `.sample()`，`step()` 接口完全没变。

`current_observations` 保留旧字典引用，`step()` 返回的新字典赋给 `observations`。这不是深拷贝，不要自行原地修改旧数组；将来写入长期训练缓冲区时，可对各数组使用 `.copy()` 保存快照。

双方不必采用同一种方法：可以让三个追捕者用神经网络，同时让逃逸者继续用规则。

## 10 分别控制双方 做四个实验

先进入项目根目录。教学脚本支持双方策略选择和 `--inspect`，暂不支持原随机脚本的 `--output`，只打印统计。

### 实验一 双方随机

```powershell
python '.\src\simple_tag_policy_demo.py' --pursuers random --evader random --episodes 2
```

对应原随机演示，核对两者动作生成的位置。参数和种子相同时，两者使用相同的动作采样设种子方法。

### 实验二 追捕者追踪 逃逸者不主动行动

```powershell
python '.\src\simple_tag_policy_demo.py' --pursuers chase --evader stay --episodes 2
```

观察红色智能体是否趋向绿色。绿色仍可能被碰撞推动，不是位置固定的障碍物。

### 实验三 追踪与逃离

```powershell
python '.\src\simple_tag_policy_demo.py' --pursuers chase --evader flee --episodes 2
```

只换逃逸规则。观察追捕者是否出现追在后面、撞上障碍物或缺少封堵等问题。

### 实验四 单独观察逃逸规则

```powershell
python '.\src\simple_tag_policy_demo.py' --pursuers stay --evader flee --episodes 2
```

追捕者不主动施加控制，便于看清逃逸者如何选择最近威胁；物理碰撞和阻尼仍存在。

想快速运行，追加 `--render none`；想查看首步，追加 `--inspect`：

```powershell
python '.\src\simple_tag_policy_demo.py' --pursuers chase --evader flee --render none --episodes 10 --max-cycles 100 --seed 42
```

少量教学回合不能作为策略优劣的正式结论。比较方法要共享初始条件、步数上限和统计口径。

## 11 接入自己的规则 从改几行开始

先尝试在 `select_action()` 的追捕者 `chase` 分支中，只根据横向位置差行动：

```python
# 用以下代码替换 chase 分支原有的方向计算和返回。
dx = observation[12]
if abs(dx) < 1e-8:
    return 0
return 2 if dx > 0 else 1
```

这个规则故意忽略纵向，不保证抓到目标。目的是核对：修改的几行是否真的影响追捕方，而不影响逃逸方。

理解后可读取 `obs[4:8]` 构造避障方向，或利用 `obs[8:12]` 的队友位置减少重复追踪。得到障碍物信息，不等于环境会自动避障，具体决策仍要实现。

修改默认逃逸规则时，继续进入 `flee_action()`，其中边界感知已实现；旧的纯远离逻辑只用于 `--unbounded`。

若新增策略名如 `avoid`，要同时增加命令行 `choices` 和 `select_action()` 的对应分支。只添加命令行选项不会自动实现行为。

无论内部规则多复杂，最终都必须为每个当前智能体返回合法动作。不要漏掉逃逸者，不要用下一时刻观测生成当前动作，也不要让策略直接读取 `env.unwrapped.world` 绕过规定的观测权限。

## 12 从规则到学习策略 还缺哪些部分

### 12.1 推理与训练是两件事

规则策略是观测经过手写计算输出动作；模型推理是观测经过已有参数的网络输出动作。二者接入动作字典的位置相同。

可以让追捕者共享一个 actor：每个追捕者输入自己的 16 维观测，得到五个动作的分数或概率，再按算法采样或在评估时选择动作。不能将五个分数直接交给当前离散环境，要先得到动作编号。

以下只是表达替换关系的伪代码，不可原样执行，项目中尚不存在 `actor` 或 `choose_action`：

```python
if agent.startswith("adversary_"):
    action = actor.choose_action(observations[agent])
else:
    action = evader_rule(observations[agent])
actions[agent] = int(action)
```

只有推理不会自动训练。还需要经验收集、训练目标和参数更新。把 `--episodes` 改成很大、不断打印奖励，并不会让策略变聪明。

### 12.2 一条训练数据如何对齐

一次交互可记为 `(obs_t, action_t, reward_t, obs_t_plus_1, terminated, truncated)`。这里 `reward_t` 表示执行时刻 t 的动作后收到的奖励。

1. 在 `step()` 前保存旧观测，基于它生成当前动作。
2. 提交动作组合。
3. 保存返回的本步奖励、下一观测、终止和截断标志。
4. 按需要收集追捕方用于参数更新的训练数据，逃逸动作仍必须生成。
5. 达到算法规定的收集长度后，计算训练目标并更新追捕方参数。

PPO/MAPPO 还需要旧动作概率、价值估计等信息，它们不是环境自动给出的。任务终止与时间截断对价值自举的处理也应依据算法区分，不能直接不加判断地合并。

### 12.3 集中训练与分散执行

执行时，每个追捕者的 actor 读取自身允许使用的观测。共享同一函数或网络参数，不等于自然产生协作。

训练时，可让集中式 critic 使用团队或全局信息。`env.state()` 在当前配置返回 `(62,)`；本机实现按名称顺序拼接各智能体观测，包含重复信息，不是 62 个互相独立的物理变量。

如果需要当前和下一全局状态，要分别在 `step()` 前后读取，保持时间对齐。执行时不要把训练专用的全局信息输入本应只读取自身观测的 actor。

训练算法、神经网络、优化器、模型保存和训练脚本仍需后续开发，本次教学示例未实现这些功能。

## 13 深入时 按顺序读库源码

先运行下面的命令定位当前环境中的 MPE2 安装目录，再在 VSCode 中打开对应文件并搜索函数名。以下路径中的 `<Python安装目录>` 是占位符，需要替换为你的实际安装目录；非 Windows 系统的目录结构可能不同。

```powershell
python -c "from pathlib import Path; import mpe2; print(Path(mpe2.__file__).resolve().parent)"
```

先读场景，最后读物理实现；这些文件先阅读，不直接修改。

### 第一层 场景定义

```text
<Python安装目录>\Lib\site-packages\mpe2\simple_tag\simple_tag.py
```

按顺序读 `raw_env.__init__()`、`Scenario.make_world()`、`reset_world()`、`observation()`、`adversary_reward()`、`agent_reward()`、`is_collision()`、`is_terminal()`。

对应实体配置、初始状态、观测拼接、奖励和成功定义。你可以自己推导观测为什么是 16 和 14 维，以及为什么追捕者共享接触奖励。

### 第二层 动作映射与环境步进

```text
<Python安装目录>\Lib\site-packages\mpe2\_mpe_utils\simple_env.py
```

看 `_set_action()` 如何将整数变成控制，`_execute_world_step()` 如何推进世界并计算奖励，`step()` 如何检查步数与终止。

该场景底层是 AEC 实现，再转换成 Parallel 接口。AEC 依次收集各智能体动作，本场景收齐一轮后统一更新物理世界。先坚持使用本项目的 `parallel_env()` 和动作字典，不要混入 AEC 的 `agent_iter()`、`last()` 和单动作提交方式。

### 第三层 物理计算

```text
<Python安装目录>\Lib\site-packages\mpe2\_mpe_utils\core.py
```

看 `World.step()`、`apply_action_force()`、`apply_environment_force()`、`integrate_state()` 和 `get_collision_force()`，理解控制力、碰撞、位置与速度。读懂它们不是写第一个策略的前置条件。

## 14 常见疑问

**为什么装了库还不会追捕？** 环境负责执行任务，动作由你的策略提供，随机采样不会自动变成学习策略。

**逃逸方为什么叫 `agent_0`？** 这是本场景的命名约定，红色追捕者叫 `adversary_*`，绿色逃逸者叫 `agent_0`。

**观测为什么不是图片？** 当前使用场景给出的数值向量；图像是另一种输入方案，需要另外取像素并处理。

**为什么原随机示例没有墙？** 按协作约定保留原示例作环境参考；有限场地在规则演示中默认启用。

**所有追捕者奖励相同，是否代表协作？** 不代表。这是默认团队奖励的计算方式。

**为什么不施加动作仍会移动？** 速度和碰撞响应不会因动作 0 被取消。

**看得到障碍物为什么还会撞？** 教学规则没有使用障碍物信息；避障需要另写规则或学习。

**改了参数为什么切片失效？** 固定索引依赖数量和排序，先核对 `observation()` 和实际形状。

**相同种子为什么换策略后轨迹不同？** 初始场景相同，但动作不同会产生不同后续状态。自己的随机源也需设种子。

**VSCode 为什么使用了错误的 Python？** 使用 “Python: Select Interpreter” 选择安装了本项目依赖的解释器；此前记住的解释器可能需要手动切换。

## 15 Python 环境与依赖复现

当前示例已在 Windows、Python 3.13.12 上验证。仓库不绑定个人安装目录，也不会自动创建 `.venv`。请先确认 `python` 指向你打算使用的解释器。

```powershell
python -c "import sys; from importlib.metadata import version; print(sys.executable); print('mpe2', version('mpe2'))"
python -m pip check
```

如需在 PowerShell 中显式指定解释器，可使用下面的格式，并将占位路径替换为实际路径：

```powershell
& '<你的Python路径>\python.exe' '.\src\simple_tag_demo.py' --episodes 2
```

正常学习不用重复安装。Windows 下仅缺少依赖时可执行以下脚本；它默认使用 PATH 中的 `python`，也支持 `-PythonExecutable` 指定解释器路径：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

需要复现锁定版本时可用 `python -m pip install -r requirements-lock.txt`。这会按记录安装依赖；如果所选 Python 同时被其他项目使用，应先确认是否存在不同版本要求。

## 16 学完后的自测

1. 能否说出创建环境、重置、双方决策、提交动作、统计和关闭资源的顺序？
2. 能否解释原随机脚本为什么取得观测却没有使用它？
3. 能否写一个动作字典，为四个智能体指定不同动作？
4. 能否从观测中找到逃逸者方向，以及逃逸者眼中最近的追捕者？
5. 能否指出自己的规则应改哪个函数，而不修改库？
6. 能否区分本步奖励、累计奖励、成功和超时？
7. 加入正的距离奖励后，为什么成功判断需要改变？
8. 能否区分规则决策、模型推理和训练更新？

理解这些以后，再做避障规则基线、学习基线、改进模块和正式对照实验。课程基础版只训练追捕方，固定逃逸规则。正式实验还需要统一条件、分开训练验证测试种子，并报告多次独立训练的稳定性。

## 官方资料与版本依据

- [MPE2 Simple Tag](https://mpe2.farama.org/environments/simple_tag/)：场景和参数概览。
- [PettingZoo Parallel API](https://pettingzoo.farama.org/api/parallel/)：动作字典与返回值规范。
- [PettingZoo AEC API](https://pettingzoo.farama.org/api/aec/)：顺序收集动作的另一套接口，初学本项目时不必混用。
- [MPE2 官方仓库](https://github.com/Farama-Foundation/MPE2)：源码与版本记录。

官方网页可能随版本更新。本说明精确的切片、团队奖励、物理更新顺序和最后一步截断行为，以第 13 节列出的本机 1.1.1 源码为准。
