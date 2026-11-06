# 协作开发说明

本仓库从 MPE2 Simple Tag 的教学示例起步。当前有随机动作和简单追踪、逃离规则，后续在此基础上实现课程项目。建议先读 README 和 MPE2 库与演示动画说明，再修改代码。

## 获取项目与运行

使用 Python 3.11 或以上版本，目前验证环境为 Windows 和 Python 3.13.12。克隆后在仓库根目录执行：

```powershell
git clone https://github.com/zacharyczy/mpe2-simple-tag-tutorial.git
cd mpe2-simple-tag-tutorial
python -m pip install -r requirements.txt
python src/simple_tag_demo.py --render none --episodes 3
python src/simple_tag_policy_demo.py --pursuers chase --evader flee --render none --episodes 3
```

已安装相应依赖时可直接运行，不需要重复安装。也可以使用自己的 Python 环境；不要提交环境目录，不要在脚本中写死个人电脑的安装路径。

## 开发流程

仓库拥有者在 GitHub 的 Settings → Collaborators 中邀请成员，成员接受邀请后才有直接推送分支的权限。没有写入权限时，也可以 Fork 后提交 Pull Request。

开始任务前先确定分工，尽量用 Issue 记录任务目标、验收方式和负责人。

```powershell
git switch main
git pull --ff-only
git switch -c feature/your-change
```

完成一个清晰的小改动后，先运行相关示例，再选择实际修改的文件提交：

```powershell
git add src/simple_tag_policy_demo.py README.md
git commit -m "Describe the concrete change"
git push -u origin feature/your-change
```

在 GitHub 上创建 Pull Request，写清改了什么、如何运行验证、是否改变观测/动作/奖励/终止或评价条件。合并前由另一位成员阅读和检查。

## 修改约定

- 新规则先接入 `select_action()` 的对应分支；新增策略名称时同步修改命令行 `choices` 和文档。
- 保留原随机示例作为环境检查参考。它不是学习基线，也没有训练神经网络。
- 修改智能体数量、障碍物数量或观测限制时，重新核对 16/14 维观测及固定切片。
- 如修改奖励，重新核对 `capture_success()`；目前它依赖单逃逸者和默认接触奖励的约定。
- 记录种子和实验配置。比较方法应使用共同测试初始条件，训练、验证、测试种子分开。
- 新增依赖写入 `requirements.txt`，并说明用途；统一依赖版本时再更新 `requirements-lock.txt`，不要把个人环境的全部包直接导出到项目中。
- 不直接修改 `site-packages` 中的库文件；自定义逻辑放在项目里，必要时封装环境。
- 不提交运行日志、模型大文件、虚拟环境、编辑器个人设置、口令或访问凭据。当前 `.gitignore` 已排除常见的本地文件。

后续实现强化学习时，分别组织策略、环境配置、训练、评估和结果记录。文档中的伪代码不是已有功能，新增功能后应同步更新实现状态。

## 上游来源

本项目调用 [Farama Foundation MPE2](https://github.com/Farama-Foundation/MPE2)，使用 [PettingZoo](https://pettingzoo.farama.org/) 接口。库源码通过依赖安装，不复制进本仓库。报告中请按课程要求说明参考项目来源，以及本组实际完成的改进与实验。
