"""
基于 ResNet-101 的鳍 aspect 分类模块 (PyTorch 版本)
支持训练和推理
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2
import numpy as np
import os
import random
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from concurrent.futures import ThreadPoolExecutor


# ============================================
# 1. 构建 ResNet-101 Fin Aspect 分类模型
# ============================================

CLASS_NAMES = ['DL', 'DR', 'MISC']


class FinAspectModel(nn.Module):
    """基于 ResNet101 的 Fin Aspect 分类模型 (DL/DR/MISC)"""

    def __init__(self, num_classes=3, pretrained=True):
        super(FinAspectModel, self).__init__()

        # 加载预训练 ResNet101
        self.base_model = models.resnet101(weights='IMAGENET1K_V1' if pretrained else None)

        # 获取原全连接层的输入特征数
        in_features = self.base_model.fc.in_features

        # 替换为新的分类头
        self.base_model.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.5),
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # 处理 BatchNorm1d 在 batch_size=1 时的问题
        if self.training and x.size(0) == 1:
            self.eval()
            out = self.base_model(x)
            self.train()
            return out
        return self.base_model(x)

    def freeze_base(self):
        """冻结 ResNet101 底层，只训练分类头"""
        # 冻结所有参数
        for param in self.base_model.parameters():
            param.requires_grad = False
        # 解冻分类头 (fc层) 的参数
        for param in self.base_model.fc.parameters():
            param.requires_grad = True

    def unfreeze_base(self):
        """解冻所有层"""
        for param in self.base_model.parameters():
            param.requires_grad = True


def build_fin_aspect_model(num_classes=3, pretrained=True):
    """
    构建 Fin Aspect 分类模型
    返回: (model, device)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FinAspectModel(num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)
    return model, device


# ============================================
# 2. 数据集和数据增强
# ============================================

class FinAspectDataset(Dataset):
    """Fin Aspect 分类数据集，目录结构为 data_dir/DL, data_dir/DR, data_dir/MISC"""

    def __init__(self, data_dir=None, transform=None, samples=None):
        """
        Args:
            data_dir: 数据目录，假设结构为 data_dir/DL/, data_dir/DR/, data_dir/MISC/
            transform: 图像变换
            samples: 可选，显式的 (img_path, label) 列表；提供时忽略 data_dir 扫描，
                     用于 train/val 拆分后复用同一 Dataset 类
        """
        self.data_dir = data_dir
        self.transform = transform

        if samples is not None:
            self.samples = samples
        else:
            self.samples = []
            self._load_samples()

    def _load_samples(self):
        """加载样本路径"""
        for label, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.exists(class_dir):
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        self.samples.append((os.path.join(class_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # 读取图像
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def split_dataset(data_dir, val_ratio=0.2, seed=42):
    """
    按类别分层拆分数据集为训练/验证集
    返回: (train_samples, val_samples)，元素为 (img_path, label)
    """
    dataset = FinAspectDataset(data_dir=data_dir)
    rng = random.Random(seed)

    by_class = {}
    for sample in dataset.samples:
        by_class.setdefault(sample[1], []).append(sample)

    train_samples, val_samples = [], []
    for label, samples in by_class.items():
        samples = samples.copy()
        rng.shuffle(samples)
        val_count = max(1, int(len(samples) * val_ratio))
        val_samples.extend(samples[:val_count])
        train_samples.extend(samples[val_count:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def get_transforms(train=True):
    """获取数据变换；训练时不做水平翻转 (DL/DR 为镜像类别)"""
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def create_data_loaders(data_dir, val_ratio=0.2, batch_size=32, num_workers=4, seed=42):
    """
    创建训练和验证数据加载器 (自动分层拆分)
    """
    train_samples, val_samples = split_dataset(data_dir, val_ratio=val_ratio, seed=seed)

    train_dataset = FinAspectDataset(
        samples=train_samples,
        transform=get_transforms(train=True)
    )

    val_dataset = FinAspectDataset(
        samples=val_samples,
        transform=get_transforms(train=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


# ============================================
# 3. 完整的训练流程
# ============================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc


def compute_class_weights(samples, num_classes=3, device=None):
    """根据训练集样本数计算逆频率类别权重"""
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, label in samples:
        counts[label] += 1
    weights = counts.sum() / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_model(data_dir, val_ratio=0.2, epochs=30, batch_size=32, num_workers=4,
                lr_phase1=1e-3, lr_phase2=1e-5, save_dir='./models', seed=42):
    """
    完整的训练流程
    """
    os.makedirs(save_dir, exist_ok=True)

    # 创建模型
    model, device = build_fin_aspect_model()

    # 数据加载器
    train_loader, val_loader = create_data_loaders(
        data_dir, val_ratio, batch_size, num_workers, seed
    )

    # 逆频率类别权重，防止 MISC 类 (样本极少) 被淹没
    train_samples = train_loader.dataset.samples
    class_weights = compute_class_weights(train_samples, device=device)
    print(f"类别权重 ({'/'.join(CLASS_NAMES)}): "
          f"{[round(float(w), 4) for w in class_weights]}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 第一阶段：冻结 ResNet101 底层，只训练分类头
    print("阶段1：训练分类头...")
    model.freeze_base()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=lr_phase1)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
    )

    best_acc = 0.0
    patience_counter = 0
    patience = 5

    history1 = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(min(epochs, 10)):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history1['train_loss'].append(train_loss)
        history1['train_acc'].append(train_acc)
        history1['val_loss'].append(val_loss)
        history1['val_acc'].append(val_acc)

        print(f"Epoch [{epoch+1}/10] Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'fin_aspect_resnet101_phase1.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停于 epoch {epoch+1}")
                break

    # 第二阶段：微调整个网络
    print("\n阶段2：微调整个网络...")
    model.unfreeze_base()
    optimizer = optim.Adam(model.parameters(), lr=lr_phase2)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
    )

    best_acc = 0.0
    patience_counter = 0

    history2 = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history2['train_loss'].append(train_loss)
        history2['train_acc'].append(train_acc)
        history2['val_loss'].append(val_loss)
        history2['val_acc'].append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'fin_aspect_resnet101_final.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停于 epoch {epoch+1}")
                break

    return model, history1, history2


# ============================================
# 4. 推理和预测
# ============================================

class FinAspectClassifier:
    """
    Fin Aspect 分类器封装类
    """

    def __init__(self, model_path=None, device=None, num_workers=8):
        self.input_size = (224, 224)
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Use", self.device)
        self.class_names = CLASS_NAMES

        # 创建模型
        self.model, _ = build_fin_aspect_model(pretrained=(model_path is None))

        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Load Aspect model: {model_path}")
        else:
            print(f"Aspect model isn't not valid: {model_path}")

        self.model.eval()

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # 预处理线程池常驻复用，避免每个 batch 反复创建/销毁线程
        self._pool = ThreadPoolExecutor(max_workers=num_workers) if num_workers > 1 else None

    def close(self):
        """关闭内部线程池。"""
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _to_pil(self, image_input):
        """将多种输入类型转换为 PIL Image (RGB)"""
        if isinstance(image_input, str):
            return Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            # OpenCV / numpy array: BGR -> RGB
            if image_input.ndim == 2:
                # 灰度图
                return Image.fromarray(image_input).convert('RGB')
            elif image_input.ndim == 3 and image_input.shape[2] == 3:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                return Image.fromarray(img_rgb)
            elif image_input.ndim == 3 and image_input.shape[2] == 4:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGRA2RGB)
                return Image.fromarray(img_rgb)
            else:
                return Image.fromarray(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            return image_input.convert('RGB')
        else:
            raise TypeError(f"不支持的图像输入类型: {type(image_input)}")

    def preprocess(self, image_input):
        """预处理单张图像
        Args:
            image_input: 图像路径(str)、PIL Image 或 numpy array
        """
        img = self._to_pil(image_input)
        img = self.transform(img)
        img = img.unsqueeze(0)  # 增加 batch 维度
        return img.to(self.device)

    def predict(self, image_input):
        """
        预测单张图像
        Args:
            image_input: 图像路径(str)、PIL Image 或 numpy array
        返回: (类别, 置信度, 概率分布)
        """
        img = self.preprocess(image_input)

        with torch.no_grad():
            outputs = self.model(img)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        class_idx = predicted.item()
        probs = probabilities[0].cpu().numpy()

        return {
            'class': self.class_names[class_idx],
            'class_idx': class_idx,
            'confidence': float(confidence.item()),
            'probabilities': {
                self.class_names[i]: float(probs[i])
                for i in range(len(self.class_names))
            }
        }

    def predict_batch(self, image_inputs, num_workers=None):
        """批量预测
        Args:
            image_inputs: 图像路径/PIL Image/numpy array 的列表
            num_workers: 已废弃，预处理线程数由构造参数 num_workers 决定，
                线程池常驻复用（保留此参数仅为兼容旧调用）
        Returns:
            list[dict]: 每张图像的预测结果，格式与 predict() 一致
        """
        # 预处理（PIL 转换 + resize + 归一化）是 CPU 密集操作，多线程并行
        if self._pool is not None and len(image_inputs) > 1:
            images = list(self._pool.map(lambda item: self.transform(self._to_pil(item)), image_inputs))
        else:
            images = [self.transform(self._to_pil(item)) for item in image_inputs]

        images = torch.stack(images).to(self.device)

        with torch.no_grad():
            outputs = self.model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)

        results = []
        for i, (pred, conf) in enumerate(zip(predicted, confidences)):
            probs = probabilities[i].cpu().numpy()
            result = {
                'class': self.class_names[pred.item()],
                'class_idx': pred.item(),
                'confidence': float(conf.item()),
                'probabilities': {
                    self.class_names[j]: float(probs[j])
                    for j in range(len(self.class_names))
                }
            }
            # 保留路径信息（如果输入是字符串）
            if isinstance(image_inputs[i], str):
                result['path'] = image_inputs[i]
            else:
                result['index'] = i
            results.append(result)
        return results


# ============================================
# 5. 评估和可视化
# ============================================

def evaluate_model(model_path, data_dir, val_ratio=0.2, batch_size=32, num_workers=4, seed=42):
    """
    评估模型性能并生成报告
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model, _ = build_fin_aspect_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 创建测试数据加载器
    _, val_samples = split_dataset(data_dir, val_ratio=val_ratio, seed=seed)
    test_dataset = FinAspectDataset(samples=val_samples, transform=get_transforms(train=False))
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total

    print(f"验证集损失: {test_loss:.4f}")
    print(f"验证集准确率: {test_acc:.2f}%")

    # 生成分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds,
                               target_names=CLASS_NAMES))

    print("\n混淆矩阵:")
    print(confusion_matrix(all_labels, all_preds))

    return all_labels, all_preds


# ============================================
# 6. 使用示例
# ============================================

if __name__ == "__main__":
    # 示例1：快速测试模型结构
    print("构建模型...")
    model, device = build_fin_aspect_model()
    print(f"模型设备: {device}")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 示例2：训练（取消注释以使用）
    # model, hist1, hist2 = train_model(
    #     data_dir='/media/dolphin/PICWD/QinZhou/CropJPEGImages_LRSplit',
    #     epochs=30,
    #     batch_size=32
    # )

    # 示例3：推理
    # classifier = FinAspectClassifier('models/fin_aspect_resnet101_final.pth')
    # result = classifier.predict('./test_image.jpg')
    # print(f"预测结果: {result['class']}, 置信度: {result['confidence']:.2%}")
