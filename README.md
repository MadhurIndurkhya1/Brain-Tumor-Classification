# 🧠 Brain Tumor Classification

A Deep Learning-based **Brain Tumor Classification** system that uses Convolutional Neural Networks (CNNs) to classify brain MRI images into four different categories.

The project is implemented using **Python, TensorFlow/Keras, OpenCV, NumPy, and Scikit-learn** and is developed as an end-to-end image classification workflow.

---

## 📌 Overview

Brain tumor classification from MRI scans is an important computer vision application in the healthcare domain.

This project uses a custom **Convolutional Neural Network (CNN)** to analyze MRI images and classify them into:

* 🧠 **Glioma Tumor**
* 🧠 **Meningioma Tumor**
* ✅ **No Tumor**
* 🧠 **Pituitary Tumor**

The complete workflow includes:

**MRI Images → Image Preprocessing → Dataset Preparation → Train/Test Split → CNN Training → Validation → Prediction**

---

## 🚀 Features

* Classification of brain MRI images into **4 categories**
* Image resizing and preprocessing using OpenCV
* Dataset shuffling for better training distribution
* Train/test dataset splitting
* One-hot encoding of target labels
* Custom CNN architecture built using Keras
* Dropout layers for reducing overfitting
* Adam optimizer
* Categorical Cross-Entropy loss
* Accuracy-based model evaluation
* Training and validation performance visualization

---

## 🧠 Model Architecture

The project uses a custom CNN architecture built with TensorFlow/Keras.

### Architecture

```text
Input MRI Image
     │
     ▼
150 × 150 × 3
     │
     ▼
Conv2D (32 filters)
     │
     ▼
Conv2D (64 filters)
     │
     ▼
MaxPooling + Dropout
     │
     ▼
Conv2D (64 filters)
     │
     ▼
Conv2D (64 filters)
     │
     ▼
MaxPooling + Dropout
     │
     ▼
Conv2D (128 filters)
     │
     ▼
Conv2D (128 filters)
     │
     ▼
Conv2D (128 filters)
     │
     ▼
MaxPooling + Dropout
     │
     ▼
Conv2D (128 filters)
     │
     ▼
Conv2D (256 filters)
     │
     ▼
MaxPooling + Dropout
     │
     ▼
Flatten
     │
     ▼
Dense (512)
     │
     ▼
Dense (512)
     │
     ▼
Dropout
     │
     ▼
Dense (4) + Softmax
     │
     ▼
Predicted Tumor Class
```

The model contains approximately **4.45 million trainable parameters**.

---

## 📊 Dataset

The project uses the **Brain Tumor Classification MRI dataset**.

The dataset contains four classes:

| Class              | Description                 |
| ------------------ | --------------------------- |
| `glioma_tumor`     | Glioma tumor MRI images     |
| `meningioma_tumor` | Meningioma tumor MRI images |
| `no_tumor`         | MRI images without a tumor  |
| `pituitary_tumor`  | Pituitary tumor MRI images  |

Images are resized to:

```text
150 × 150 × 3
```

The notebook processes **3,264 MRI images** after loading the dataset.

---

## ⚙️ Technologies Used

| Technology       | Purpose                          |
| ---------------- | -------------------------------- |
| Python           | Programming language             |
| TensorFlow       | Deep learning framework          |
| Keras            | CNN model development            |
| OpenCV           | Image processing                 |
| NumPy            | Numerical computation            |
| Scikit-learn     | Dataset splitting and evaluation |
| Matplotlib       | Visualization                    |
| Seaborn          | Data visualization               |
| Jupyter Notebook | Development environment          |

---

## 🔄 Project Workflow

### 1. Dataset Loading

MRI images are loaded from the training and testing directories.

### 2. Image Preprocessing

Each image is:

* Read using OpenCV
* Resized to `150 × 150`
* Converted into a NumPy array

### 3. Dataset Preparation

The images and corresponding labels are shuffled before splitting.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_train,
    Y_train,
    test_size=0.1,
    random_state=101
)
```

### 4. Label Encoding

The four tumor categories are converted into numerical labels and then transformed using one-hot encoding.

### 5. CNN Training

The CNN is trained for **10 epochs** using:

* **Optimizer:** Adam
* **Loss:** Categorical Cross-Entropy
* **Metric:** Accuracy

### 6. Evaluation

Training and validation accuracy/loss are monitored throughout the training process.

---

## 📈 Results

The model's validation performance improved throughout training.

|  Epoch | Training Accuracy | Validation Accuracy |
| -----: | ----------------: | ------------------: |
|      1 |            28.04% |              33.67% |
|      2 |            29.36% |              27.89% |
|      3 |            37.76% |              41.16% |
|      4 |            53.12% |              52.04% |
|      5 |            58.87% |              51.70% |
|      6 |            68.18% |              55.78% |
|      7 |            70.79% |              59.52% |
|      8 |            73.36% |              60.88% |
|      9 |            77.07% |              79.93% |
| **10** |        **81.50%** |          **85.37%** |

The highest recorded validation accuracy in the notebook is **85.37%** at epoch 10.

> **Note:** This is the validation accuracy reported during training, not a claim of clinical diagnostic accuracy.

---

## 📂 Repository Structure

```text
Brain-Tumor-Classification/
│
├── brain-tumor-classification.ipynb
│
└── README.md
```

The main notebook contains the complete implementation, including dataset loading, preprocessing, CNN construction, training, and evaluation.

---

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/MadhurIndurkhya1/Brain-Tumor-Classification.git
```

Navigate into the project:

```bash
cd Brain-Tumor-Classification
```

Install the required Python libraries:

```bash
pip install tensorflow keras numpy opencv-python scikit-learn matplotlib seaborn pillow tqdm
```

---

## ▶️ Running the Project

The project is implemented as a Jupyter Notebook.

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
brain-tumor-classification.ipynb
```

Run the notebook cells sequentially.

### Dataset Path

The notebook was developed using the Kaggle dataset environment and expects the dataset structure approximately as:

```text
brain-tumor-classification-mri/
│
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
│
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/
```

---

## 🔬 Key Machine Learning Concepts Demonstrated

This project demonstrates practical implementation of:

* Convolutional Neural Networks
* Image classification
* Image preprocessing
* Feature extraction through convolution layers
* Max pooling
* Dropout regularization
* Dense neural networks
* Softmax classification
* One-hot encoding
* Train/validation splitting
* Model training and evaluation
* Training/validation performance analysis

---

## 🔮 Future Improvements

Possible improvements for the project include:

* [ ] Data augmentation
* [ ] Transfer learning using ResNet, EfficientNet, or MobileNet
* [ ] Hyperparameter optimization
* [ ] Class-wise precision, recall, and F1-score
* [ ] Confusion matrix
* [ ] Grad-CAM / Explainable AI visualization
* [ ] Model checkpointing
* [ ] Early stopping
* [ ] Model deployment using Streamlit or Gradio
* [ ] REST API for model inference
* [ ] Docker-based deployment

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**.

It should **not be used as a substitute for professional medical diagnosis, clinical decision-making, or medical advice**. MRI classification models can make errors and require appropriate clinical validation before any real-world medical application.

---

## 👨‍💻 Author

**Madhur Indurkhya**

B.Tech — Computer Science & Engineering
Specialization: Artificial Intelligence & Machine Learning

### 🔗 Links

* **GitHub:** [MadhurIndurkhya1](https://github.com/MadhurIndurkhya1)
* **Project Repository:** [Brain-Tumor-Classification](https://github.com/MadhurIndurkhya1/Brain-Tumor-Classification)

---

## ⭐ Acknowledgements

This project was developed as a practical application of **Deep Learning and Computer Vision** techniques for medical image classification.

If you find this project useful, consider giving the repository a ⭐ on GitHub.
