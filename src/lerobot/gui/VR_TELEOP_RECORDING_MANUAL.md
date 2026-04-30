# VR 遥操作采集数据操作手册

本文档说明如何使用 `src/lerobot/gui/hil_recording` 提供的图形界面进行 Quest3 VR 遥操作和 Human-in-Loop 数据采集。界面名称为 **Evo-RL HIL Recorder**，主窗口标题区显示 **Ola Data Recorder**。

## 1. 适用场景

该 GUI 主要用于以下流程：

- 使用 Quest3 VR 控制双臂 PiPER 机器人采集示教数据。
- 选择纯人工遥操作，或接入 OpenPI/本地策略进行 Human-in-Loop 采集。
- 在采集过程中通过按钮或快捷键标记成功、失败、重录、切换人工/策略控制。
- 采集前检查相机、CAN、ADB/Quest3 连接状态。

## 2. 启动前检查

### 2.1 软件环境

在采集电脑上进入项目目录，并使用项目要求的 conda 环境：

```bash
cd /home/ola/code/Evo-RL
conda activate evo-rl-test
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
```

启动 GUI：

```bash
python -m lerobot.gui.hil_recording.main
```

如果已安装项目入口命令，也可以使用：

```bash
lerobot-human-inloop-record-gui
```

### 2.2 硬件连接

启动录制前确认：

- 机器人已上电，左右臂处于可控制状态。
- CAN 接口已连接，默认接口名为 `can_left` 和 `can_right`。
- Quest3 已连接到电脑，`adb devices` 能看到状态为 `device` 的设备。
- 相机已连接。录制脚本支持 `CAMERA_PROFILE=none`、`realsense`、`gopro`。
- 如果使用 OpenPI Remote，确认 OpenPI 目录存在，并且能运行 `scripts/serve_policy.py`。

常用检查命令：

```bash
adb devices
```

如需指定相机配置，可在启动 GUI 前设置：

```bash
export CAMERA_PROFILE=realsense
# 或
export CAMERA_PROFILE=gopro
```

## 3. 界面区域说明

GUI 分为左侧参数区和右侧操作区。

### 3.1 顶部状态

顶部右侧状态 badge 显示当前录制状态：

| 状态 | 含义 |
| --- | --- |
| `IDLE` | 空闲，尚未开始录制 |
| `STARTING` | 录制进程正在启动 |
| `RECORDING` | 正在采集 episode |
| `RESETTING` | episode 结束后正在复位环境 |
| `STOPPED` | 录制进程已结束 |
| `ERROR` | 启动或运行过程中出错 |

录制运行中，左侧参数会被锁定，避免采集中误改配置。

### 3.2 Dataset 参数

左侧 **Dataset** 区域用于设置数据集：

| 参数 | 说明 |
| --- | --- |
| `Dataset Name` | 数据集 ID，格式通常为 `用户名/数据集名` |
| `Task` | 本批数据的任务描述，会写入数据集 |
| `Num Episodes` | 需要采集的 episode 数量 |
| `Episode Time (s)` | 单个 episode 最长录制时间 |
| `Reset Time (s)` | 两个 episode 之间的环境复位时间 |
| `Resume Existing Dataset` | 勾选后继续写入已有本地数据集 |

本地数据集默认保存在 LeRobot 的缓存目录下，例如：

```text
~/.cache/huggingface/lerobot/<Dataset Name>
```

如果 `Dataset Name` 对应的本地目录已经存在，并且没有勾选 `Resume Existing Dataset`，GUI 会弹窗提供三个选项：

- **删除并重新录制**：删除已有本地数据集后重新开始。
- **添加 --resume**：自动勾选 Resume，继续追加采集。
- **取消并改名**：取消启动，回到参数区修改数据集名称。

### 3.3 Policy 参数

左侧 **Policy** 区域用于选择采集模式：

| Policy Mode | 适用情况 | 需要填写 |
| --- | --- | --- |
| `Teleop Only` | 只用 VR 人工遥操作采集 | 不需要策略参数 |
| `OpenPI Remote` | GUI 自动启动 OpenPI server，录制进程通过 host/port 调用策略 | `OpenPI Policy Dir`、`OpenPI Server Root`、`Policy Host`、`Policy Port` |
| `Local Policy Path` | 直接加载本地策略路径 | `Policy Path` |

OpenPI Remote 模式下，GUI 会在 `OpenPI Server Root` 目录中启动：

```text
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_bipiper_absolute_lora --policy.dir=<OpenPI Policy Dir> --port=<Policy Port>
```

录制进程随后连接到 `Policy Host:Policy Port`。

## 4. 采集流程

### 4.1 启动 GUI

```bash
cd /home/ola/code/Evo-RL
conda activate evo-rl-test
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
python -m lerobot.gui.hil_recording.main
```

如需采集 RealSense 画面：

```bash
export CAMERA_PROFILE=realsense
python -m lerobot.gui.hil_recording.main
```

### 4.2 填写参数

1. 在 `Dataset Name` 填入本次数据集名称。
2. 在 `Task` 写清楚本次任务目标。
3. 设置 `Num Episodes`、`Episode Time (s)` 和 `Reset Time (s)`。
4. 选择 `Policy Mode`：
   - 纯人工采集选择 `Teleop Only`。
   - 策略加人工接管选择 `OpenPI Remote` 或 `Local Policy Path`。
5. 如果要继续采已有数据集，勾选 `Resume Existing Dataset`。

GUI 会自动保存上一次填写的参数到：

```text
~/.config/evo-rl/hil_recording_gui.json
```

下次打开会自动恢复。

### 4.3 自检

点击右侧 **自检** 按钮，观察 **Status Monitor**：

| 项目 | 检查内容 |
| --- | --- |
| `Camera` | 检查配置中的目标相机是否在线 |
| `CAN Status` | 检查 `can_left`、`can_right`，必要时尝试自动激活 |
| `ADB Devices` | 检查 Quest3/Android 设备是否通过 ADB 连接 |

建议三个状态均为 `OK` 后再开始录制。若为 `WARNING` 或 `ERROR`，先看卡片下方说明，再查看底部 **Process Log**。

### 4.4 开始录制

点击 **开始录制**。

GUI 会执行以下动作：

1. 检查数据集路径是否冲突。
2. 保存当前参数。
3. 如果是 OpenPI Remote 模式，先启动 OpenPI server。
4. 启动录制脚本：

```text
src/lerobot/gui/hil_recording/run_teleop_with_vt3_ik.sh
```

录制开始后状态变为 `RECORDING`。底部 **Process Log** 会显示启动命令、PID、录制进度和错误信息。

## 5. VR 操作方式

Quest3 控制器默认按键如下。

### 5.1 右臂

| Quest3 按键 | 作用 |
| --- | --- |
| 按住 `B` | 启用右臂 VR 控制。松开后保持当前位置 |
| `A` | 将右臂控制基准重置到初始姿态 |
| 右扳机 `rightTrig` | 在右臂启用时切换夹爪开/合 |

### 5.2 左臂

| Quest3 按键 | 作用 |
| --- | --- |
| 按住 `Y` | 启用左臂 VR 控制。松开后保持当前位置 |
| `X` | 将左臂控制基准重置到初始姿态 |
| 左扳机 `leftTrig` | 在左臂启用时切换夹爪开/合 |

### 5.3 操作建议

- 先让双手控制器保持稳定，再按住 `B` 或 `Y` 启用对应手臂。
- 移动过快可能导致 IK 拒绝当前目标，建议动作连续、平滑。
- 如果手臂和控制器相对位置不自然，松开启用键，调整手柄姿态后再重新按住启用键。
- episode 之间进入 `RESETTING` 时，按界面提示把物体和环境恢复到初始状态。

## 6. GUI 录制控制按钮和快捷键

录制过程中可使用右侧按钮，也可直接按键盘快捷键。

| 按钮 | 快捷键 | 作用 |
| --- | --- | --- |
| `Success` | `S` | 标记当前 episode 成功，并结束当前 episode |
| `Fail` | `F` | 标记当前 episode 失败，并结束当前 episode |
| `重录` | 左方向键 | 结束当前 episode，并清空本 episode 缓存后重录 |
| `Human/Policy` | `I` | 在策略控制和人工接管之间切换 |
| `结束当前阶段` | 右方向键 | 提前结束当前录制或复位阶段 |
| `停止录制` | `Esc` | 停止整个录制任务 |
| `自检` | 无 | 检查相机、CAN、ADB 状态 |

说明：

- `Human/Policy` 只有在同时启用策略和 VR 遥操作时有意义。
- 在 `Teleop Only` 模式下，采集始终由人工 VR 控制。
- `Success` 和 `Fail` 会写入 episode 结果标签，并结束当前 episode。
- `停止录制` 会先发送 `Esc` 给录制进程；如果进程未及时退出，GUI 会尝试强制结束。

## 7. 完成采集

当达到 `Num Episodes`，或点击 **停止录制** 后，状态会变为 `STOPPED`。

建议完成后检查：

1. **Process Log** 中是否出现异常。
2. 本地数据集目录是否生成。
3. episode 数量是否符合预期。

可使用数据集报告命令做快速检查：

```bash
lerobot-dataset-report --dataset <Dataset Name>
```

## 8. 日志和配置文件

GUI 日志文件：

```text
~/.config/evo-rl/hil_recording_gui.log
```

GUI 参数缓存：

```text
~/.config/evo-rl/hil_recording_gui.json
```

底部 **Process Log** 中能看到：

- 实际启动的录制命令。
- OpenPI server 启动和退出信息。
- 录制进程 PID。
- episode 录制和 reset 提示。
- 失败原因和异常输出。

## 9. 常见问题

### 9.1 ADB Devices 显示 `No ADB devices`

处理步骤：

1. 检查 Quest3 是否通过 USB 连接。
2. 在 Quest3 中允许 USB 调试授权。
3. 运行：

```bash
adb devices
```

确认设备状态为 `device`，不要是 `unauthorized` 或空列表。

### 9.2 CAN Status 报错

GUI 会尝试运行项目中的 CAN 激活脚本：

```text
scripts/can_muti_activate.sh --ignore
```

如果仍失败，检查：

- CAN 设备是否插好。
- 接口名是否为 `can_left`、`can_right`。
- 机器人是否上电。
- 当前用户是否有权限操作 CAN。

### 9.3 Camera 报 Missing

检查：

- 是否设置了正确的 `CAMERA_PROFILE`。
- RealSense 序列号是否和脚本中的配置一致。
- GoPro `/dev/video*` 路径是否和环境变量一致。

GoPro 默认路径：

```text
GOPRO_LEFT_WRIST_INDEX_OR_PATH=/dev/video0
GOPRO_RIGHT_WRIST_INDEX_OR_PATH=/dev/video2
GOPRO_RIGHT_FRONT_INDEX_OR_PATH=/dev/video4
```

### 9.4 OpenPI Remote 启动失败

检查：

- `OpenPI Server Root` 是否存在，默认是 `/home/ola/code/openpi`。
- 该目录下是否有 `scripts/serve_policy.py`。
- `OpenPI Policy Dir` 是否指向有效 checkpoint。
- `Policy Port` 是否被其他进程占用。

### 9.5 数据集已经存在

如果不想覆盖旧数据，选择 **取消并改名**，修改 `Dataset Name`。

如果要继续采集同一数据集，选择 **添加 --resume**。

如果确认旧数据不需要，选择 **删除并重新录制**。

### 9.6 VR 没有控制机器人

检查：

- `ADB Devices` 自检是否 OK。
- Quest3 companion app 是否被自动启动。
- Process Log 是否有 `[VR_HEALTH]` 相关错误。
- 是否按住了对应手臂的启用键：右臂 `B`，左臂 `Y`。
- 如果使用策略模式，是否需要点击 `Human/Policy` 或按 `I` 切到人工接管。

## 10. 推荐现场流程

1. 上电机器人，连接 CAN、相机、Quest3。
2. `conda activate evo-rl-test` 后启动 GUI。
3. 填写 Dataset 和 Policy 参数。
4. 点击 **自检**，确认 Camera/CAN/ADB 正常。
5. 点击 **开始录制**。
6. 在 Quest3 中按住 `B`/`Y` 控制左右臂，使用扳机控制夹爪。
7. 每个 episode 完成后点击 `Success` 或 `Fail`。
8. 需要重来时点击 **重录**。
9. 全部采完后等待状态变为 `STOPPED`，再检查数据集。
