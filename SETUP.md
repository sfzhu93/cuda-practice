# CUDA 开发环境配置

## 前提条件

- NVIDIA GPU，驱动已安装（`nvidia-smi` 可用）
- Python 3.13
- pip

## 步骤

### 1. 创建 venv

```bash
python3 -m venv .venv
```

### 2. 安装 CUDA 工具链和 clangd

```bash
source .venv/bin/activate
pip install nvidia-cuda-nvcc clangd
```

这会安装：
- `nvidia-cuda-nvcc` — nvcc 编译器
- `nvidia-cuda-runtime` — libcudart
- `nvidia-nvvm` — libdevice（PTX 后端）
- `nvidia-cuda-crt` — CUDA C 运行时头文件
- `clangd` — LSP 服务器（用于 VSCode 代码补全）

安装后关键路径：
```
.venv/lib/python3.13/site-packages/nvidia/cu13/bin/nvcc
.venv/lib64/python3.13/site-packages/clangd/data/bin/clangd
```

### 3. 确认 GPU 架构

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

对照表（常见型号）：

| 架构 | 代表型号 | `sm_` |
|------|---------|-------|
| Blackwell | RTX 5070/5080/5090 | `sm_120` |
| Ada Lovelace | RTX 4070/4080/4090 | `sm_89` |
| Ampere | RTX 3070/3080/3090 | `sm_86` |
| Turing | RTX 2080 | `sm_75` |

按实际 GPU 修改 `Makefile` 中的 `GPU_ARCH`，以及 `.clangd` 中的 `--cuda-gpu-arch`。

### 4. 编译运行

```bash
make run
```

## Makefile 说明

`Makefile` 中的关键变量：

```makefile
VENV     := ./.venv
CUDA_DIR := $(VENV)/lib/python3.13/site-packages/nvidia/cu13
NVCC     := $(CUDA_DIR)/bin/nvcc
CUDA_LIB := $(CUDA_DIR)/lib
GPU_ARCH := sm_120   # ← 按你的 GPU 修改
```

新增 `.cu` 文件后，直接在 `Makefile` 的 `TARGETS` 里加上文件名（不含扩展名）即可。

## VSCode clangd 配置

项目已包含 `.vscode/settings.json` 和 `.clangd`，克隆后无需额外配置。

若路径不对（换了机器或 Python 版本），更新 `.vscode/settings.json`：

```json
{
  "clangd.path": "/path/to/cuda_practice/.venv/lib64/python3.13/site-packages/clangd/data/bin/clangd"
}
```

确认真实 clangd 路径：
```bash
find .venv -name clangd -type f
```

配置完在 VSCode 执行 **Ctrl+Shift+P → "clangd: Restart Language Server"**。

> **注意**：`.venv/bin/clangd` 是 Python 包装脚本，`lib64/.../clangd/data/bin/clangd` 才是真正的二进制，VSCode 需要指向后者。

## Nsight Compute (ncu) profiling

```bash
make profile              # profile gemm（默认）
make profile PROFILE=vec_add
```

生成 `profile_<name>.ncu-rep`，在 ncu-ui 里 **File → Open** 打开。

### 权限配置（一次性）

`ncu` 需要访问 GPU performance counters，默认仅 root 可用。永久开放：

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
```

重启后生效。验证：`cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly`（应为 `0`）。

> 注意：不要写进 `/etc/modprobe.d/nvidia.conf`，ASUS ROG 电源管理工具会覆盖该文件。

生效前临时用 root 跑：`make profile`（Makefile 里已加 `sudo`）。

## 常见问题

**`nvidia-smi` 有输出但 `nvcc` 找不到**
驱动和 CUDA 工具链是分开的。驱动装好后，nvcc 需要单独安装，pip 方案是最简单的方式。

**编译报 `sm_XXX` 不支持**
nvcc 版本太旧不认识新架构。升级：`pip install -U nvidia-cuda-nvcc`。

**运行时报 `libcudart.so` 找不到**
Makefile 已通过 `-Xlinker -rpath,...` 把库路径硬编码进可执行文件，直接 `./vec_add` 即可。若手动编译时忘了加，用：
```bash
export LD_LIBRARY_PATH=.venv/lib/python3.13/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
```
