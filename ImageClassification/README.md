# ImageClassification

一个通用图像分类学习框架，支持 5 种数据集和 5 种网络结构，提供训练、可视化、推理、Web 应用完整链路。

## 文件结构

```
ImageClassification/
├── train.py       训练脚本（数据增强 / AdamW / cosine / 早停 / 双 ckpt / CSV 日志）
├── infer.py       单图推理（配置驱动，加载权重后对一张图做预测）
├── draw.py        训练曲线看板（读 CSV，画 loss/acc/lr/best 四张图）
├── app.py         FastAPI Web 应用（模型服务 + 前端页面）
├── index.html     Web 前端（纯原生，零 CDN，拖拽上传，动态渲染）
├── dataset.py     5 个数据集类（MNIST/CIFAR10/CIFAR100/OCT/Garbage）
├── model.py       5 个标准网络（LeNet5/AlexNet8/VGGNet16/GoogLeNet/ResNet18）
└── util.py        工具函数（train_once/test_once/Checkpoint/EpochLogger）
```

## 前置条件

```powershell
# 激活项目 venv
.\.venv\Scripts\Activate.ps1

# 安装依赖
pip install torch torchvision pandas matplotlib fastapi uvicorn python-multipart torch-summary
```

## 数据集

数据放在项目根目录 `Datasets/` 下。当前默认用 Garbage 分类数据：

```
Datasets/GarbageClassification/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

ImageFolder 按子文件夹名自动识别类别，无需额外标注文件。

## 使用

所有命令在项目根目录 `LearnDeep/` 下运行。

### 1. 训练

```powershell
python ImageClassification/train.py
```

输出：
- `Checkpoints/best_image_classification_model.pth`（test loss 最低时的权重）
- `Checkpoints/last_image_classification_model.pth`（最后一个 epoch 的权重）
- `Logs/image_classification_log.csv`（每 epoch 的 train/test loss、acc、lr）

超参数在 `train.py` 的 `Config` 里改：`EPOCHS`、`BATCH_SIZE`、`LEARNING_RATE`、`WEIGHT_DECAY`、`PATIENCE`、`SEED`。

### 2. 查看训练曲线

```powershell
python ImageClassification/draw.py
```

读 `Logs/image_classification_log.csv`，画 4 张图存到 `Logs/plots/training_curves.png`。`Config.SHOW = True` 时弹窗，`False` 时存图。

### 3. 单图推理

```powershell
python ImageClassification/infer.py
```

对 `Config.IMAGE_PATH` 指定的图片做预测，打印最可能的类别。

### 4. Web 应用

```powershell
python ImageClassification/app.py
```

浏览器打开 `http://127.0.0.1:8000/`，拖拽图片即可识别。

接口：
- `GET /` → 前端页面
- `GET /api/classes` → 返回类别列表
- `POST /api/predict` → 上传图片，返回 top-5 概率

## 数据集类（dataset.py）

每个数据集类实现 4 个方法：

| 方法 | 说明 |
|------|------|
| `get_train_dataset()` | 返回训练集（用 `train_transforms`） |
| `get_test_dataset()` | 返回测试集（用 `test_transforms`） |
| `get_class_names()` | 返回类别名列表 |
| `get_channel_nums()` | 返回通道数（1=灰度，3=RGB） |

构造参数统一为 `(train_transforms, test_transforms, data_dir, ...)`，`train_transforms`/`test_transforms` 传 `None` 表示不做增强（推理场景）。

## 网络（model.py）

5 个网络都继承 `BaseImageClassificationModel`，构造签名统一：

```python
ModelName(input_size=(C, H, W), num_classes=N)
```

| 网络 | 默认输入 | 参数量级 |
|------|---------|---------|
| LeNet5 | (1, 28, 28) | ~60K |
| AlexNet8 | (3, 227, 227) | ~61M |
| VGGNet16 | (3, 224, 224) | ~138M |
| GoogLeNet | (3, 224, 224) | ~6M |
| ResNet18 | (3, 224, 224) | ~11M |

## 切换数据集 / 网络

训练时只需改 `train.py` 里两处：

```python
# 数据集
dataset = CIFAR10ImageClassificationDataset(
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    data_dir=Config.CIFAR_10_DATASET_PATH,
)

# 网络
model = VGGNet16(input_size=(channels, 224, 224), num_classes=num_classes).to(Config.DEVICE)
```

`channels` 和 `num_classes` 自动从 dataset 拿，不用手写。

## 早停机制

`Checkpoint.update()` 每 epoch 调用一次：
- test loss 创新低 → 存 `best.pth`，返回 `(True, False)`
- 连续 `PATIENCE` 个 epoch 没创新低 → 返回 `(False, True)`，训练循环 break
- 每 epoch 都存 `last.pth`

## Web 应用说明（app.py + index.html）

- `app.py` 用 `InferenceService` 类封装模型（只加载一次，不每请求重新 load）
- `index.html` 纯原生 HTML + CSS，零 CDN 依赖，类别颜色用 HSL 色相环自动生成（任意 N 类适配）
- 换数据集时改 `app.py` 顶部 `CKPT_PATH` / `DATA_DIR` 即可，前端不用动
