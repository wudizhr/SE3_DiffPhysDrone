# util 目录说明

`util/` 用来存放跨训练、可视化、模型、环境都会复用的轻量工具。

当前不建议把已有模块强行搬进这里，因为项目已经有更明确的公共模块：

- `se3diff_config/`：配置、路径解析、Env 构造。
- `visualization/visualization_common.py`：可视化导出公共函数。
- `visualization/rviz2_common.py`：RViz2 播放公共函数。
- `train/loss/`：训练损失。

适合以后放入 `util/` 的内容：

- `seed.py`：统一设置 random、torch、CUDA 随机种子。
- `device.py`：选择 cuda/cpu、检查设备可用性。
- `tensor.py`：轻量 tensor/numpy 转换和 shape 检查。
- `checkpoint.py`：查找最新 checkpoint、检查 checkpoint 目录。
- `logging.py`：通用打印格式，不包含 TensorBoard 业务逻辑。

不建议放入 `util/` 的内容：

- 配置加载和合并。
- Env 构造。
- 可视化 marker、npz、图像处理。
- 训练 loss。
- 暂时不知道放哪里的杂项代码。

经验规则：同一个小函数被两个以上子系统重复使用时，再移动到 `util/`。
