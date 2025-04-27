# food_recognition

# Food-101 图像分类项目

## 项目概述
本项目使用 Food-101 数据集实现食物图像分类任务，包含两种不同的实现方式：
1. 使用 PyTorch 和 ResNet50 从头开始训练模型
2. 使用 Hugging Face 提供的预训练 ViT 模型(nateraw/food)进行迁移学习

## 数据集
使用 Hugging Face 提供的 Food-101 数据集：
```python
from datasets import load_dataset
ds = load_dataset("ethz/food101")
```
- 包含 101 类食物图像
- 训练集：75,750 张图像
- 测试集：25,250 张图像
- 每类有 750 张训练图像和 250 张测试图像

## 方法一：ResNet50 训练

### 实现步骤
1. 数据预处理与增强
2. 加载预训练 ResNet50 模型
3. 替换最后的全连接层(适配 101 类)
4. 模型训练与评估

### 关键代码
```python
# 数据预处理
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载模型
model = models.resnet50(weights="IMAGENET1K_V2")
model.fc = nn.Linear(model.fc.in_features, 101)
```

## 方法二：Hugging Face 预训练模型

### 模型信息
使用 [nateraw/food](https://huggingface.co/nateraw/food) 预训练模型：
- 基于 google/vit-base-patch16-224-in21k 微调
- 在 Food-101 测试集上的表现：
  - Loss: 0.4501
  - Accuracy: 0.8913

### 使用方式
```python
from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("nateraw/food")
model = AutoModelForImageClassification.from_pretrained("nateraw/food")
```

### 优势
- 开箱即用，无需训练
- 准确率更高(约 89.13%)
- 支持批量推理

## 项目结构
```
food-classification/
├── resnet50_train.py       # ResNet50 训练代码
├── huggingface_inference.py # 预训练模型推理代码
├── utils.py               # 工具函数
├── README.md              # 项目说明
└── requirements.txt       # 依赖列表
```


## 使用示例
1. ResNet50 训练：
```bash
python resnet50_train.py
```

2. 预训练模型推理：
```bash
python huggingface_inference.py --image_path your_image.jpg
```

## 未来改进
- 尝试更多数据增强技术
- 对 ResNet50 进行超参数调优
- 实现模型集成(ResNet50 + ViT)
- 开发 Web 演示界面

## 参考资源
- [Food-101 数据集](https://huggingface.co/datasets/ethz/food101)
- [nateraw/food 模型](https://huggingface.co/nateraw/food)
- [PyTorch 官方教程](https://pytorch.org/tutorials/)