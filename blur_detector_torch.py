"""
基于 ResNet-101 的图像模糊检测模块 (PyTorch 版本)
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
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import copy


# ============================================
# 1. 构建 ResNet-101 模糊检测模型
# ============================================

class BlurDetectionModel(nn.Module):
    """基于 ResNet101 的模糊检测模型"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super(BlurDetectionModel, self).__init__()
        
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


def build_blur_detection_model(num_classes=2, pretrained=True):
    """
    构建模糊检测模型
    返回: (model, device)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BlurDetectionModel(num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)
    return model, device


# ============================================
# 2. 数据集和数据增强
# ============================================

class BlurDataset(Dataset):
    """模糊检测数据集"""
    
    def __init__(self, data_dir, transform=None, blur_augment=False, blur_ratio=0.5):
        """
        Args:
            data_dir: 数据目录，假设结构为 data_dir/clear/ 和 data_dir/blur/
            transform: 图像变换
            blur_augment: 是否对清晰图像进行模糊增强
            blur_ratio: 模糊增强比例
        """
        self.data_dir = data_dir
        self.transform = transform
        self.blur_augment = blur_augment
        self.blur_ratio = blur_ratio
        self.gaussian_kernels = [(5, 5), (7, 7), (9, 9), (15, 15)]
        self.motion_kernels = self._create_motion_kernels()
        
        self.samples = []
        self._load_samples()
    
    def _create_motion_kernels(self):
        """创建不同方向的运动模糊核"""
        kernels = []
        for size in [7, 11, 15]:
            # 水平运动
            kernel_h = np.zeros((size, size))
            kernel_h[int((size-1)/2), :] = np.ones(size)
            kernel_h = kernel_h / size
            
            # 垂直运动
            kernel_v = np.zeros((size, size))
            kernel_v[:, int((size-1)/2)] = np.ones(size)
            kernel_v = kernel_v / size
            
            kernels.extend([kernel_h, kernel_v])
        return kernels
    
    def _load_samples(self):
        """加载样本路径"""
        # 清晰图像: 标签 0
        clear_dir = os.path.join(self.data_dir, 'clear')
        if os.path.exists(clear_dir):
            for fname in os.listdir(clear_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.samples.append((os.path.join(clear_dir, fname), 0))
        
        # 模糊图像: 标签 1
        blur_dir = os.path.join(self.data_dir, 'blur')
        if os.path.exists(blur_dir):
            for fname in os.listdir(blur_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.samples.append((os.path.join(blur_dir, fname), 1))
    
    def apply_random_blur(self, image):
        """随机应用一种模糊效果"""
        import random
        img_array = np.array(image)
        blur_type = random.choice(['gaussian', 'motion', 'defocus'])
        
        if blur_type == 'gaussian':
            kernel_size = random.choice(self.gaussian_kernels)
            sigma = random.uniform(1.5, 4.0)
            blurred = cv2.GaussianBlur(img_array, kernel_size, sigma)
            
        elif blur_type == 'motion':
            kernel = random.choice(self.motion_kernels)
            blurred = cv2.filter2D(img_array, -1, kernel)
            
        else:  # defocus
            radius = random.randint(3, 7)
            blurred = cv2.medianBlur(img_array, radius * 2 + 1)
        
        return Image.fromarray(blurred)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 读取图像
        image = Image.open(img_path).convert('RGB')
        
        # 模糊增强：如果是清晰图像，有一定概率进行模糊
        if self.blur_augment and label == 0 and np.random.random() < self.blur_ratio:
            image = self.apply_random_blur(image)
            label = 1
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(train=True):
    """获取数据变换"""
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
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


def create_data_loaders(train_dir, val_dir, batch_size=32, num_workers=4):
    """
    创建训练和验证数据加载器
    """
    train_dataset = BlurDataset(
        train_dir, 
        transform=get_transforms(train=True),
        blur_augment=True,
        blur_ratio=0.5
    )
    
    val_dataset = BlurDataset(
        val_dir,
        transform=get_transforms(train=False),
        blur_augment=False
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


def train_model(train_dir, val_dir, epochs=20, batch_size=32, num_workers=4, 
                lr_phase1=1e-3, lr_phase2=1e-5, save_dir='./models'):
    """
    完整的训练流程
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建模型
    model, device = build_blur_detection_model()
    
    # 数据加载器
    train_loader, val_loader = create_data_loaders(
        train_dir, val_dir, batch_size, num_workers
    )
    
    criterion = nn.CrossEntropyLoss()
    
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
            torch.save(model.state_dict(), os.path.join(save_dir, 'blur_detection_resnet101_phase1.pth'))
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
            torch.save(model.state_dict(), os.path.join(save_dir, 'blur_detection_resnet101_final.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停于 epoch {epoch+1}")
                break
    
    return model, history1, history2


# ============================================
# 4. 推理和预测
# ============================================

class BlurDetector:
    """
    模糊检测器封装类
    """
    def __init__(self, model_path=None, device=None):
        self.input_size = (224, 224)
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['clear', 'blur']
        
        # 创建模型
        self.model, _ = build_blur_detection_model(pretrained=(model_path is None))
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"已加载模型: {model_path}")
        
        self.model.eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
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
    
    def predict_batch(self, image_inputs):
        """批量预测
        Args:
            image_inputs: 图像路径/PIL Image/numpy array 的列表
        """
        images = []
        for item in image_inputs:
            img = self._to_pil(item)
            img = self.transform(img)
            images.append(img)
        
        images = torch.stack(images).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)
        
        results = []
        for i, (pred, conf) in enumerate(zip(predicted, confidences)):
            result = {
                'class': self.class_names[pred.item()],
                'confidence': float(conf.item())
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

def evaluate_model(model_path, test_dir, batch_size=32, num_workers=4):
    """
    评估模型性能并生成报告
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model, _ = build_blur_detection_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 创建测试数据加载器
    test_dataset = BlurDataset(test_dir, transform=get_transforms(train=False))
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
    
    print(f"测试集损失: {test_loss:.4f}")
    print(f"测试集准确率: {test_acc:.2f}%")
    
    # 生成分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, 
                               target_names=['清晰', '模糊']))
    
    print("\n混淆矩阵:")
    print(confusion_matrix(all_labels, all_preds))
    
    return all_labels, all_preds


# ============================================
# 6. 使用示例
# ============================================

if __name__ == "__main__":
    # 示例1：快速测试模型结构
    print("构建模型...")
    model, device = build_blur_detection_model()
    print(f"模型设备: {device}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 示例2：训练（取消注释以使用）
    # model, hist1, hist2 = train_model(
    #     train_dir='./dataset/train',
    #     val_dir='./dataset/val',
    #     epochs=25,
    #     batch_size=16
    # )
    
    # 示例3：推理
    # detector = BlurDetector('models/blur_detection_resnet101_final.pth')
    # result = detector.predict('./test_image.jpg')
    # print(f"预测结果: {result['class']}, 置信度: {result['confidence']:.2%}")
