# 配套 StarIR 单卡训练代码

本目录保存停止训练时的服务器代码，网络与损失保持官方实现；
单进程验证已修复，随机颜色增强仅用于训练，兼容当前 SciPy 导入路径。

对应的当前权重、恢复状态、配置及停止记录位于上一层。
详细设置和恢复说明见 ../README.md。

使用环境：Python 3.12.3，PyTorch 2.8.0+cu128，CUDA 12.8；
torchvision 0.23.0+cu128、einops、PyYAML、opencv-python-headless、lmdb、
scipy、scikit-image、tqdm、tensorboard。
