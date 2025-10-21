# 🌾 Rice Leaf Disease Classification & Detection

A deep learning project for automated classification and detection of rice leaf diseases using state-of-the-art convolutional neural networks (CNNs). This project implements multiple architectures including ResNet18, Inception ResNet V2, and VGG19 to classify three major rice diseases.

## 📊 Dataset Information

**Dataset Source**: [Rice Leaf Diseases Dataset](https://share.google/bL3EzdTbEWDqk6yLn)

### Disease Classes
The dataset contains images of three major rice leaf diseases:
- **Bacterial Leaf Blight**: 40 images
- **Brown Spot**: 40 images  
- **Leaf Smut**: 41 images

**Total Images**: 121 images across 3 disease classes

### Dataset Split
- **Training**: 96 images (80%)
- **Validation**: 12 images (10%)
- **Testing**: 12 images (10%)



## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SyedAejazAhmed/Rice-Leaf-Classification-Detection.git
   cd "Rice Leaf Classification & Detection"
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   # Install basic dependencies
   pip install -r req.txt
   
   # Install CUDA dependencies (for GPU support)
   pip install -r cuda.txt
   ```

### Dataset Preparation

1. **Download the dataset** from [this link](https://share.google/bL3EzdTbEWDqk6yLn)

2. **Run the data splitting script**
   ```bash
   cd rice_leaf_diseases
   python datasplit_corrected.py
   ```
   
   This will create the `dataset/` folder with train/val/test splits.

### Training Models

#### Option 1: ResNet18
```bash
cd ResNet18
jupyter notebook resnet18.ipynb
```

#### Option 3: VGG19
```bash
cd vgg19
jupyter notebook vgg19.ipynb
```

## 🧠 Model Architectures

We tested three different deep learning models to classify rice leaf diseases:

### 1. ResNet18 🏗️
**What is ResNet18?**
ResNet18 is a convolutional neural network that's like a smart image analyzer with 18 layers. It's designed to recognize patterns in images by learning from millions of photos.

**How we used it:**
- Started with a pre-trained model (already knows how to recognize basic image features)
- Modified the final layer to recognize our 3 disease types instead of 1000 general categories
- Used transfer learning (borrowed knowledge from other image tasks)

**Results:**
- Achieved 50% accuracy on test images
- Took 38 training cycles to reach best performance

### 2. VGG19 🔍
**What is VGG19?**
VGG19 is a deep convolutional neural network with 19 layers that's known for its simplicity and effectiveness. It uses small 3×3 filters throughout the network, making it excellent at capturing fine details in images. Think of it as a meticulous detective that examines every small detail.

**How we used it:**
- Started with a pre-trained VGG19 model (trained on millions of images)
- Froze the feature extraction layers to preserve learned patterns
- Modified the classifier to output 3 disease categories instead of 1000
- Added dropout layers to prevent overfitting

**Results:**
- Training in progress...
- Expected to perform well on detailed image analysis

### 3. Inception ResNet V2 🏆
**What is Inception ResNet V2?**
This is a more advanced neural network that combines two powerful techniques: Inception (which looks at images in multiple ways simultaneously) and ResNet (which helps information flow better through the network). Think of it as a more sophisticated image detective.

**How we used it:**
- Started with a pre-trained model that already understands complex image patterns
- Adapted it specifically for our rice disease classification task
- Added dropout layers to prevent overfitting (memorizing instead of learning)

**Results:**
- Achieved perfect 100% accuracy on test images
- Reached optimal performance in just 10 training cycles
- **Winner of our comparison! 🥇**

## 📈 Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|-------|----------|-----------|---------|----------|---------|
| **Inception ResNet V2** | **100%** | **100%** | **100%** | **100%** | **1.0** |
| ResNet18 | 50% | 33.3% | 50% | 40% | 0.75 |
| VGG19 | *Training...* | *TBD* | *TBD* | *TBD* | *TBD* |

## 🛠️ Key Features

### Data Preprocessing
- **Image Augmentation**: Random resizing, horizontal flips, normalization
- **Reproducible Splits**: Fixed random seed (42) for consistent results
- **Balanced Dataset**: Equal representation across disease classes

### Training Approach
- **Smart Learning**: Used pre-trained models that already know basic image features
- **Optimization**: Used advanced algorithms to help models learn efficiently
- **Validation**: Split data to test model performance on unseen images
- **Best Model Selection**: Saved the model that performed best during training

### Performance Evaluation
- **Accuracy**: How often the model makes correct predictions
- **Precision**: When model predicts a disease, how often is it right?
- **Recall**: Out of all actual disease cases, how many did the model catch?
- **F1-Score**: A balanced measure combining precision and recall
- **ROC Curves**: Visual representation of model performance

## 📋 Requirements

### Required Software
- **Python**: Programming language used for the project
- **PyTorch**: Deep learning framework for building neural networks
- **Jupyter Notebook**: Interactive environment for running the models
- **Various Libraries**: For data processing, visualization, and model evaluation

## 📊 Training Process

### Training Setup
- **Batch Processing**: Models learned from 32 images at a time
- **Learning Speed**: Set to learn gradually to avoid mistakes
- **Training Cycles**: Up to 100 rounds of learning (stopped early if performance plateaued)
- **Image Processing**: All images resized to 224×224 pixels for consistency

### Data Preparation
- **Image Standardization**: All images made the same size
- **Data Augmentation**: Created variations (rotations, flips) to help models generalize
- **Normalization**: Adjusted image brightness and colors for better learning

## 🏆 Results Analysis

### Inception ResNet V2 (Winner 🥇)
- **Perfect Classification**: 100% accuracy on test set
- **Robust Performance**: No misclassifications
- **Fast Convergence**: Best model at epoch 10

### ResNet18
- **Baseline Performance**: 50% accuracy
- **Learning Curve**: Gradual improvement over 38 epochs
- **Overfitting Issues**: High training accuracy, lower test performance

### VGG19
- **Deep Architecture**: 19 layers for detailed feature extraction
- **Fine Detail Analysis**: Excellent at capturing small patterns in leaf images
- **Transfer Learning**: Leverages pre-trained ImageNet features
- **Status**: Currently in training phase

## 🔧 Technical Notes
- **GPU Acceleration**: Models can run faster with NVIDIA graphics cards
- **Transfer Learning**: We used pre-trained models to save training time
- **Data Augmentation**: Images were rotated and resized to improve learning

## 📝 Usage Examples

### How the Models Work
1. **Input**: Feed a rice leaf image to the model
2. **Processing**: The neural network analyzes patterns, colors, shapes, and textures
3. **Classification**: Model outputs a prediction for one of three diseases
4. **Confidence**: Each prediction comes with a confidence score

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Dataset**: Rice Leaf Diseases Dataset contributors
- **Frameworks**: PyTorch, torchvision, timm
- **Pretrained Models**: ImageNet pretrained weights
- **Community**: Open source deep learning community

## 📞 Contact

For questions, issues, or collaborations:
- Create an issue in the repository
- Contact the maintainers

---

**⭐ Star this repository if you found it helpful!**

## 🔗 Related Links

- [PyTorch Documentation](https://pytorch.org/docs/)
- [TIMM Library](https://github.com/rwightman/pytorch-image-models)
- [Rice Disease Classification Papers](https://scholar.google.com/scholar?q=rice+leaf+disease+classification)
