"""
基于 ResNet-101 的图像模糊检测模块
支持训练和推理
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import os


# ============================================
# 1. 构建 ResNet-101 模糊检测模型
# ============================================

def build_blur_detection_model(input_shape=(224, 224, 3), num_classes=2):
    """
    基于 ResNet101 的模糊检测模型
    """
    # 加载预训练 ResNet101（去掉顶层）
    base_model = ResNet101(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
        pooling='avg'  # 使用全局平均池化
    )
    
    # 构建新的分类头
    inputs = base_model.input
    x = base_model.output
    
    # 添加自定义层
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='blur_prediction')(x)
    
    model = Model(inputs, outputs, name='BlurDetection_ResNet101')
    
    return model, base_model


# ============================================
# 2. 数据生成器（包含模糊增强）
# ============================================

class BlurDataGenerator:
    """
    自定义数据生成器：从清晰图像生成模糊样本
    """
    def __init__(self, blur_ratio=0.5):
        self.blur_ratio = blur_ratio
        self.gaussian_kernels = [(5, 5), (7, 7), (9, 9), (15, 15)]
        self.motion_kernels = self._create_motion_kernels()
    
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
    
    def apply_random_blur(self, image):
        """随机应用一种模糊效果"""
        blur_type = np.random.choice(['gaussian', 'motion', 'defocus'])
        
        if blur_type == 'gaussian':
            kernel_size = np.random.choice(self.gaussian_kernels)
            sigma = np.random.uniform(1.5, 4.0)
            blurred = cv2.GaussianBlur(image, kernel_size, sigma)
            
        elif blur_type == 'motion':
            kernel = np.random.choice(self.motion_kernels)
            blurred = cv2.filter2D(image, -1, kernel)
            
        else:  # defocus
            radius = np.random.randint(3, 8)
            blurred = cv2.medianBlur(image, radius * 2 + 1)
            
        return blurred
    
    def generate_training_pair(self, clear_image_path):
        """
        生成训练对：清晰图像和对应的模糊版本
        返回: (image, label)  0=清晰, 1=模糊
        """
        # 读取图像
        img = cv2.imread(clear_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        
        # 随机决定是否模糊
        if np.random.random() < self.blur_ratio:
            img = self.apply_random_blur(img)
            label = 1  # 模糊
        else:
            label = 0  # 清晰
            
        # 归一化
        img = img.astype(np.float32) / 255.0
        
        return img, label


# ============================================
# 3. 完整的训练流程
# ============================================

def create_data_generators(train_dir, val_dir, batch_size=32):
    """
    创建训练和验证数据生成器
    """
    # 训练数据增强
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # 验证数据不增强
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # 从目录加载（假设目录结构: train/blur/, train/clear/）
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        classes=['clear', 'blur'],  # 0=清晰, 1=模糊
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        classes=['clear', 'blur'],
        shuffle=False
    )
    
    return train_generator, val_generator


def train_model(train_dir, val_dir, epochs=20, batch_size=32):
    """
    完整的训练流程
    """
    # 创建模型
    model, base_model = build_blur_detection_model()
    
    # 第一阶段：冻结 ResNet101 底层，只训练分类头
    print("阶段1：训练分类头...")
    base_model.trainable = False
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    train_gen, val_gen = create_data_generators(train_dir, val_dir, batch_size)
    
    # 回调函数
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            'blur_detection_resnet101_phase1.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # 第一阶段训练
    history1 = model.fit(
        train_gen,
        epochs=min(epochs, 10),
        validation_data=val_gen,
        callbacks=callbacks
    )
    
    # 第二阶段：微调整个网络（解冻部分层）
    print("阶段2：微调整个网络...")
    base_model.trainable = True
    
    # 使用更低的学习率进行微调
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # 低学习率
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    callbacks[2] = keras.callbacks.ModelCheckpoint(
        'blur_detection_resnet101_final.h5',
        monitor='val_accuracy',
        save_best_only=True
    )
    
    # 第二阶段训练
    history2 = model.fit(
        train_gen,
        epochs=epochs,
        initial_epoch=len(history1.history['loss']),
        validation_data=val_gen,
        callbacks=callbacks
    )
    
    return model, history1, history2


# ============================================
# 4. 推理和预测
# ============================================

class BlurDetector:
    """
    模糊检测器封装类
    """
    def __init__(self, model_path=None):
        self.input_size = (224, 224)
        
        if model_path and os.path.exists(model_path):
            self.model = keras.models.load_model(model_path)
        else:
            self.model, _ = build_blur_detection_model()
            
        self.class_names = ['清晰', '模糊']
    
    def preprocess(self, image_path):
        """预处理单张图像"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # 增加 batch 维度
        return img
    
    def predict(self, image_path):
        """
        预测单张图像
        返回: (类别, 置信度, 概率分布)
        """
        img = self.preprocess(image_path)
        
        prediction = self.model.predict(img, verbose=0)
        confidence = np.max(prediction)
        class_idx = np.argmax(prediction)
        
        return {
            'class': self.class_names[class_idx],
            'class_idx': class_idx,
            'confidence': float(confidence),
            'probabilities': {
                self.class_names[i]: float(prediction[0][i]) 
                for i in range(len(self.class_names))
            }
        }
    
    def predict_batch(self, image_paths):
        """批量预测"""
        images = np.array([self.preprocess(p)[0] for p in image_paths])
        predictions = self.model.predict(images, verbose=0)
        
        results = []
        for i, pred in enumerate(predictions):
            class_idx = np.argmax(pred)
            results.append({
                'path': image_paths[i],
                'class': self.class_names[class_idx],
                'confidence': float(np.max(pred))
            })
        return results


# ============================================
# 5. 评估和可视化
# ============================================

def evaluate_model(model, test_dir):
    """
    评估模型性能并生成报告
    """
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # 评估
    results = model.evaluate(test_generator)
    print(f"测试集损失: {results[0]:.4f}")
    print(f"测试集准确率: {results[1]:.4f}")
    print(f"精确率: {results[2]:.4f}")
    print(f"召回率: {results[3]:.4f}")
    
    # 预测并生成分类报告
    test_generator.reset()
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        print("\n分类报告:")
        print(classification_report(y_true, y_pred, 
                                   target_names=['清晰', '模糊']))
    except ImportError:
        print("安装 sklearn 以获取详细分类报告")
    
    return y_true, y_pred


# ============================================
# 6. 使用示例
# ============================================

if __name__ == "__main__":
    # 示例1：快速测试模型结构
    print("构建模型...")
    model, base_model = build_blur_detection_model()
    model.summary()
    print(f"总参数量: {model.count_params():,}")
    
    # 示例2：训练（取消注释以使用）
    # model, hist1, hist2 = train_model(
    #     train_dir='./dataset/train',
    #     val_dir='./dataset/val',
    #     epochs=25,
    #     batch_size=16
    # )
    
    # 示例3：推理
    # detector = BlurDetector('blur_detection_resnet101_final.h5')
    # result = detector.predict('./test_image.jpg')
    # print(f"预测结果: {result['class']}, 置信度: {result['confidence']:.2%}")
