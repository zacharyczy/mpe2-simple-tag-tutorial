# MPE2 库与演示动画说明

这份说明解释：MPE2 替我们做了什么，以及 Demo 中一个动作怎样变成窗口里圆圈的移动。先理解这条过程，不必通读整个库。

以下依据本机 **MPE2 1.1.1** 源码，使用当前的 3 个追捕者、1 个逃逸者、2 个障碍物配置。详细观测切片和实验命令见 [README](README.md)。

## 1 整个程序由哪些部分合作完成

**我们写的 Demo 负责组织实验和选择动作。MPE2 负责维护、推进和评价仿真世界。pygame 负责将世界画到屏幕上。**

- **MPE2**：提供多个多智能体场景，Simple Tag 是其中的追逃场景。它规定实体、运动、观测、动作、奖励和终止规则。
- **PettingZoo**：规定如何与多智能体环境交互。这里通过 Parallel API，一次提交四个智能体的动作字典，再取得各自的观测和奖励。
- **Gymnasium 的空间类型**：用 `Discrete(5)` 描述五个离散动作，用 `Box(...)` 描述观测向量的形状、数值范围和类型。空间描述不是实际动作或实际观测。
- **pygame**：创建窗口、画圆、刷新画面、处理窗口事件和限制显示速度。安装包叫 `pygame-ce`，代码里用的导入名称是 `pygame`。
- **NumPy**：处理位置、速度、观测等数值数组，也用于我们规则中的距离计算。

MPE2 已提供场景和物理过程，但我们仍需为双方选择动作。它不会自动给逃逸者安排规则，也不会自动训练神经网络。

## 2 先分清我们的文件与库文件

### 我们写的两个文件

- [src/simple_tag_demo.py](src/simple_tag_demo.py)：双方每一步随机选择动作，并统计结果。
- [src/simple_tag_policy_demo.py](src/simple_tag_policy_demo.py)：支持切换随机、追踪、逃离和不施加主动控制等规则，默认仍是双方随机。

两者都是项目的调用示例，不是 MPE2 官方库源码。教学文件只从随机文件借用 `positive_int()` 和 `capture_success()`；导入不会运行随机文件的 `main()`，也不会打开两个仿真。

### 库里先认识三个文件

下面的链接指向上游当前源码，可能随版本更新。本说明的细节依据本机 1.1.1；阅读精确版本时，可运行 `python -c "from pathlib import Path; import mpe2; print(Path(mpe2.__file__).resolve().parent)"` 定位本地库目录，再打开对应文件。

**场景规则：** [simple_tag.py](https://github.com/Farama-Foundation/MPE2/blob/main/mpe2/simple_tag/simple_tag.py)

它回答“这是什么游戏”：创建几个人和几个障碍物、颜色是什么、观测包含什么、接触如何判定、奖励怎么算。

**环境交互和绘制：** [simple_env.py](https://github.com/Farama-Foundation/MPE2/blob/main/mpe2/_mpe_utils/simple_env.py)

它回答“怎样执行一次动作，怎样画出来”：建立动作/观测空间，接收动作，调用物理更新和场景规则，检查结束条件，绘制窗口。

**物理世界：** [core.py](https://github.com/Farama-Foundation/MPE2/blob/main/mpe2/_mpe_utils/core.py)

它保存实体的位置、速度、大小等状态，计算控制力、碰撞响应、阻尼和运动。

项目中的 `from mpe2 import simple_tag_v3` 是导入入口；库的 `simple_tag_v3.py` 再导出具体场景的 `env`、`parallel_env` 和 `raw_env`。你不用从这个入口文件寻找全部实现。

## 3 从启动程序到动画出现

```text
运行我们的 Demo
    ↓
读取参数，创建 Simple Tag 环境
    ↓
reset：初始化世界，取得初始观测
    ↓
根据观测或随机采样，给双方选择动作
    ↓
step(actions)：提交动作组合
    ↓
库把动作变成控制，更新世界的位置和速度
    ↓
计算奖励、结束信号和新观测
    ↓
human 模式下库自动绘制、刷新窗口
    ↓
Demo 统计结果；继续下一步，或结束并重置回合
```

这是一轮交互的概念顺序。底层还会依次收集各智能体动作，期间可能重绘尚未更新的位置，见第 5 节。

### 第一步 Demo 读取参数

在教学文件 `main()` 的 `parser.add_argument(...)` 中，规定命令行参数：

```powershell
python '.\src\simple_tag_policy_demo.py' --pursuers chase --evader flee
```

`argparse` 将它们变成 `args.pursuers`、`args.evader` 等变量。这里选择规则，还没有让实体移动。

### 第二步 创建环境和实体

```python
env = simple_tag_v3.parallel_env(
    num_good=1, num_adversaries=3, num_obstacles=2,
    # 其余参数略
    render_mode="human",
)
```

这是摘录，实际文件中还指定了离散动作、步数上限、成功终止和观测范围。

库中的 `Scenario.make_world()` 创建实体，并设置大小、加速度参数和最高速度等。实体有 `state.p_pos` 位置、`state.p_vel` 速度、`color` 颜色等属性；障碍物设置为不可移动。

`SimpleEnv` 建立动作与观测空间。当前 `continuous_actions=False`，于是库内部创建 `Discrete(5)`，我们不用再写一次。

绘图部分也在此准备一个默认 700×700 的 `pygame.Surface`。它可以理解为用于绘制的画布；这时不等于桌面窗口已经打开。

### 第三步 重置回合并取得观测

```python
observations, infos = env.reset(seed=seed)
```

场景的 `reset_world()` 为实体设置初始位置、速度和颜色，计数器也被重置。`Scenario.observation()` 根据当前世界状态计算观测。

返回的 `observations` 是字典，每个名称对应一个数组。例如 `observations["adversary_0"]` 就是追捕者 0 当前的 16 维观测。这个数组不是截图，而是包含位置、速度和相对位置的数值。

### 第四步 我们的规则选择动作

教学文件中的 `select_action()` 接收名称和该智能体自己的观测：

- `random`：调用动作空间的 `.sample()`。
- `chase`：追捕者从 `observation[12:14]` 取出逃逸者方向。
- `flee`：逃逸者找出最近追捕者，选择远离它的方向。
- `stay`：返回 0，不施加主动运动控制。

`direction_to_action()` 将期望方向转换成 0–4 的动作编号。这里控制的是决策，不负责画圆。

`main()` 对四个名称分别调用该函数，组成 `actions` 字典。此时只是完成动作选择，世界还没有推进。

### 第五步 库执行动作并更新世界

```python
observations, rewards, terminated, truncated, infos = env.step(actions)
```

Simple Tag 的底层按顺序收集动作，PettingZoo 的转换层让我们可以一次提交动作字典。收齐四个动作后，`SimpleEnv._execute_world_step()` 才统一推进世界。

内部主要做三件事：

1. `_set_action()` 将离散编号转成控制方向。例如 2 对应正 x 方向，4 对应正 y 方向，再乘以实体的控制加速度参数。
2. `World.step()` 合并主动控制与碰撞作用，按时间步更新物理状态，包括位置、速度、阻尼和最高速度限制。
3. 场景奖励函数计算本步奖励，环境检查终止/超时，再返回新观测等结果。

所以，**圆圈移动的直接原因是世界中的位置数值发生了变化，之后绘制代码使用了新位置。**

动作不是瞬移。当前物理实现先用原速度更新位置，再更新速度；静止时第一次向上可能先改变速度，下一步位置才明显改变。动作 0 也不会直接清除惯性。

## 4 窗口和圆圈具体怎样画出来

以下函数都在库的 `simple_env.py` 中。

### enable_render 创建真正的窗口

首次需要以 `human` 模式显示时，内部执行：

```python
self.screen = pygame.display.set_mode(self.screen.get_size())
self.clock = pygame.time.Clock()
```

前者创建桌面显示窗口，后者用于控制刷新节奏。当前 Demo 中，通常是在开始执行动作、内部首次调用渲染时打开窗口。

### draw 把世界状态画到画布上

`draw()` 主要完成：

1. 把画布清成白色，覆盖上一帧。
2. 根据实体当前的位置范围计算显示缩放。
3. 将仿真坐标转换成屏幕像素坐标，并反转 y 轴，使正 y 在画面上表现为向上。
4. 根据实体大小和显示缩放计算圆的半径。
5. 用实体颜色和半径画圆，并画边框。

实际画圆用的是：

```python
pygame.draw.circle(self.screen, entity.color * 200, (x, y), radius)
```

红色、绿色和深色来自场景给实体设置的颜色。缩放可以让跑远的实体继续出现在画面中；窗口边缘不等于仿真中的硬墙。

### render 把画布显示出来

`render()` 调用 `draw()`，然后在 `human` 模式中执行：

```python
pygame.event.pump()       # 处理窗口事件，避免窗口被认为无响应
pygame.display.flip()     # 将画好的内容显示到窗口
self.clock.tick(self.metadata["render_fps"])  # 限制显示频率
```

不断重复“更新状态、清空画布、按新位置画圆、刷新”，就形成了动画。这里没有读取预先制作的视频。

默认窗口不显示完整观测向量。策略读取的是内存里的数组，绘图读取的是实体状态；两者来源于同一个仿真世界，但用途不同。

## 5 Demo 为什么没写 render 也能显示

因为底层 `SimpleEnv.step()` 已经包含：

```python
if self.render_mode == "human":
    self.render()
```

我们只需选择 `render_mode="human"`，每次执行动作时库就会自动处理显示，不必在 Demo 中再调用一次 `render()`。

本机版本在底层逐个收集智能体动作时也会渲染；四个动作收齐后才更新一次物理世界。因此可能连续显示相同位置，再显示更新后的状态。`render_fps=10` 限制的是渲染调用节奏，不能直接理解成每秒推进十个完整世界步。

命令行 `--render none` 在 Demo 中会转换成 `render_mode=None`，于是跳过自动窗口显示。规则决策、物理更新、奖励和统计仍会执行，不需要靠窗口才能运行仿真。

## 6 Demo 里剩余代码各有什么作用

以教学文件为主，对应原随机文件理解即可：

- **导入部分**：`argparse` 处理参数；NumPy 做向量和距离计算；`simple_tag_v3` 提供环境；两个辅助函数来自另一个 Demo。
- **`positive_int()`**：检查回合数和步数参数，避免输入非正整数。
- **`for episode ...`**：组织多个回合，每回合调用 `reset()`。
- **随机种子代码**：让初始化和随机动作采样可以复现，不会让策略自动学习。
- **`while env.agents:`**：重复“读观测、选动作、提交动作”，是驱动仿真前进的主循环。
- **`returns` 和 `steps`**：累计奖励和交互步数，只负责记录。
- **`--inspect` 分支**：打印首回合第一步的数据，方便理解和调试，不参与决策。
- **`capture_success()`**：按当前单逃逸者和默认奖励约定判断成功，同时处理上限步接触的统计情况；奖励定义改变后需重新检查它。
- **`break`**：结束当前回合，随后由外层循环重置新回合。
- **最后的打印**：显示成功率和平均步数，不会更新策略参数。
- **`finally: env.close()`**：保证退出时释放资源；库内部调用 `pygame.quit()` 关闭显示相关资源。
- **文件末尾的入口判断**：直接运行该文件才调用 `main()`；被另一个文件导入时不自动开始仿真。

原随机 Demo 没有 `direction_to_action()` 和 `select_action()`，在循环中直接随机采样；它另外使用 `json` 和 `Path` 保存回合统计，用 `importlib.metadata.version` 打印库版本。保存数据不会影响窗口绘制。

## 7 用一个动作串起整条链路

假设追捕者当前观测中的逃逸者相对位置是 `[-0.0256, 0.6944]`：

1. **Demo 的策略**读到目标主要在上方，`direction_to_action()` 返回动作 4。
2. **Demo 的主循环**将它放进 `actions["adversary_0"]`，与其他三人的动作一并提交。
3. **环境的动作映射**将 4 解释为正 y 方向的主动控制。
4. **物理世界**依据控制、已有速度和碰撞等更新状态。
5. **场景规则**计算新观测、奖励以及是否接触成功。
6. **绘制代码**读取新位置，转换为像素坐标，重新画红圆。
7. **Demo**取得返回值，继续下一轮或结束回合。

想改变“追捕者怎么追”，先改项目的 `select_action()`；想改变“圆圈怎么画”，才涉及绘制代码；想改变“世界怎样运动”，则涉及物理实现。这是三个不同层次。

## 8 最小阅读和观察路线

先在教学 Demo 中找 `select_action()`、`env.reset()` 和 `env.step(actions)`，再到库中只搜索这些名字：

1. `simple_tag.py` 的 `make_world()`、`observation()`：实体和观测从哪里来。
2. `simple_env.py` 的 `_execute_world_step()`：一次完整动作如何推进世界。
3. `simple_env.py` 的 `enable_render()`、`draw()`、`render()`：窗口如何形成。
4. 最后按需看 `core.py` 的 `World.step()`、`integrate_state()`，不必先读完整物理实现。

在项目根目录运行下面的命令，可以同时看动画和终端里的首步数据：

```powershell
python '.\src\simple_tag_policy_demo.py' --pursuers chase --evader flee --episodes 1 --inspect
```

如果当前位于克隆仓库的上一级目录，先执行（其他位置请使用实际仓库路径）：

```powershell
Set-Location -LiteralPath .\mpe2-simple-tag-tutorial
```

读完后，你应能区分四件事：**Demo 选择动作，场景定义规则，物理模块改变状态，绘图模块显示状态。**
