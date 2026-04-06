**Overview**
- **Purpose:**: 快速让 AI 代码代理理解并在此代码库中高效工作 — 本仓库基于 Ultralytics 的 YOLO 实现，主要包位于 `ultralytics/`，顶层示例和工具散布在 `examples/`、`assets/`、和根目录脚本（如 `predict.py`）。
- **Big picture:**: 项目把 API（Python SDK）和 CLI 两条路径并行支持：Python 使用 `from ultralytics import YOLO`；命令行使用 `yolo TASK MODE ARGS`（入口由 `pyproject.toml` 的 `yolo = "ultralytics.cfg:entrypoint"` 配置）。权重、示例图片与实验输出分别位于 `weights/`、`ultralytics/assets/`、`runs/`（由 `ultralytics.utils.SETTINGS` 管理）。

**Quick Dev / Test Workflow**
- **Create env & install:**: 推荐使用 `conda` + editable install

```bash
conda create -n yolov13 python=3.11
conda activate yolov13
pip install -r requirements.txt
pip install -e .
```
- **Optional acceleration:**: README 指定 flash-attn wheel；Windows/torch 有版本限制（见 `pyproject.toml` 中对 `torch` 的注释，Windows 上避免 `torch==2.4.0`）。
- **Run examples / CLI:**: `python predict.py`（示例文件）或者使用 CLI：

```bash
yolo help
yolo predict model=yolov13n.pt source=ultralytics/assets/bus.jpg imgsz=640
yolo train detect data=coco8.yaml model=yolo11n.pt imgsz=32 epochs=1
```
- **Tests:**: 使用 `pytest`，测试会调用 `yolo` CLI 并可能下载权重/数据（参见 `tests/test_cli.py`）。建议运行子集以节约时间：

```bash
pip install -e .[dev]
pytest tests/test_cli.py -q
```

**Project-Specific Patterns & Conventions**
- **Config / CLI pattern:**: CLI/SDK 的参数体系基于 `ultralytics.cfg.get_cfg` / `check_cfg`，多数命令采用 `key=value` 覆盖（例如 `imgsz=320`、`epochs=1`）。参考帮助文本在 [ultralytics/cfg/__init__.py](ultralytics/cfg/__init__.py)。
- **Constants for paths:**: use `ASSETS` 和 `WEIGHTS_DIR` from `ultralytics.utils` rather than hardcoding — examples: `ultralytics/utils/__init__.py`（`ASSETS = ROOT / "assets"`，`WEIGHTS_DIR = Path(SETTINGS["weights_dir"])`）。
- **Model loading:**: prefer `YOLO('model.pt')` 或 `YOLO('model.yaml')` (see `predict.py`)，并通过 `model.predict(...)`, `model.train(...)`, `model.export(...)` 使用高层 API。
- **Downloads / hub:**: model/asset 下载流程集中在 `ultralytics/utils/downloads.py`（会尝试本地 `weights/`，否则从 GitHub assets 获取）。测试和例子会依赖此自动下载逻辑。
- **Experiment outputs:**: 训练/预测结果保存到 `runs/`（通过 `RUNS_DIR` / settings 管理），weights 存放在 `weights/`，设置可用 `yolo settings key=value` 修改（see `SettingsManager` 在 `ultralytics/utils/__init__.py`）。

**Integration Points & Where to Look First**
- **CLI entrypoint & help text:**: [pyproject.toml](pyproject.toml) and [ultralytics/cfg/__init__.py](ultralytics/cfg/__init__.py) — 有 TASK/MODE 列表与示例命令。
- **Paths / settings / defaults:**: [ultralytics/utils/__init__.py](ultralytics/utils/__init__.py) — `ASSETS`, `WEIGHTS_DIR`, `RUNS_DIR`, `SETTINGS`。
- **Model/engine:**: `ultralytics/engine/` 和 `ultralytics/models/` — 实现预测、训练、导出逻辑。
- **Tests & CI cues:**: `tests/` 下的测试展示常用 CLI 用法（e.g. `tests/test_cli.py`），测试会展示短命令样式和需要的 fixtures（GPU 条件、跳过规则等）。

**Practical Tips for AI Agents Editing Code**
- **Follow the existing abstractions:** 修改与新增功能应穿过 `ultralytics` 包的 API（`YOLO`, `engine.*`, `cfg`）而非在顶层脚本中硬编码路径。
- **Respect SETTINGS:** 不要直接写 `weights/` 或 `runs/` 绝对路径，使用 `SETTINGS` / `WEIGHTS_DIR` / `RUNS_DIR`。
- **Use CLI patterns for tests/examples:** 当添加新 CLI 子命令或改动行为，同步更新 `ultralytics/cfg` 的帮助字符串与 `tests/` 中相应用例。
- **Avoid platform assumptions:** `pyproject.toml` 显示多平台兼容判断（Windows 上 torch 例外），CI/tests 也会跳过 GPU-only 测试。

**Files to Read for Onboarding (priority order)**
- `README.md` — 项目概览与 quickstart
- `pyproject.toml` — 依赖、entrypoints、可选 extras
- `ultralytics/cfg/__init__.py` — CLI 架构、TASK/MODE 参考、`get_cfg`/`check_cfg`
- `ultralytics/utils/__init__.py` — 全局常量、`SettingsManager`、`ASSETS`/`WEIGHTS_DIR`
- `tests/test_cli.py` — 实践级 CLI 使用样例和测试约定
- `predict.py`, `vaild_predict.py`, `examples/*` — 常见使用示例

**When in doubt, run these commands locally**
```bash
# quick smoke: help + predict on sample asset
yolo help
yolo predict model=yolov13n.pt source=ultralytics/assets/bus.jpg imgsz=320

# run a fast unit test subset
pytest tests/test_cli.py -q
```

---
请审阅以上内容：我会把它保存为 `.github/copilot-instructions.md`，如需补充特定流程（例如 Docker 构建、Jetson 部署或导出细节），告诉我我将把它们加入。 
