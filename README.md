# Brain Tumor Detection from MRI using VGG16

A deep learning model that classifies brain MRI scans into four categories using transfer learning (VGG16 pretrained on ImageNet).

| Class | Description |
|-------|-------------|
| `glioma` | Glioma tumor |
| `meningioma` | Meningioma tumor |
| `pituitary` | Pituitary tumor |
| `notumor` | No tumor detected |

> **Disclaimer:** This is an educational project. It is not a medical device and must not be used for clinical diagnosis.

---

## Results

**Test accuracy: 94.75%** (1,600 test images, 400 per class)

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| glioma | 0.93 | 0.87 | 0.90 |
| meningioma | 0.92 | 0.94 | 0.93 |
| notumor | 0.95 | 1.00 | 0.97 |
| pituitary | 0.98 | 0.97 | 0.98 |
| **macro avg** | **0.95** | **0.95** | **0.95** |

Glioma has the lowest recall, and it is most often confused with meningioma. This overlap is common on this dataset.

---

## Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by Masoud Nickparvar (Kaggle).

```
Training/
  glioma/  meningioma/  notumor/  pituitary/
Testing/
  glioma/  meningioma/  notumor/  pituitary/
```

---

## Model

- **Backbone:** VGG16 (ImageNet weights, `include_top=False`)
- **Input size:** 224 x 224 x 3
- **Fine-tuning:** all layers frozen except the last convolutional layers
- **Head:** Flatten -> Dropout(0.3) -> Dense(128, ReLU) -> Dropout(0.2) -> Dense(4, softmax)
- **Loss:** sparse categorical crossentropy
- **Optimizer:** Adam
- **Preprocessing:** `vgg16.preprocess_input` (same at train and inference time)
- **Augmentation:** random brightness and contrast

---

## Setup

```bash
git clone https://github.com/Ridima28/Brain-Tumor-Detection-system
cd Brain-Tumor-Detection-system
pip install tensorflow numpy pillow scikit-learn matplotlib seaborn kaggle
```

Download the dataset with your Kaggle API credentials:

```bash
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
unzip brain-tumor-mri-dataset.zip
```

---

## Usage

### Train and evaluate

Open and run `Brain_Tumor_Detection.ipynb` top to bottom (Google Colab with a GPU runtime is recommended).

### Predict on a single image

```python
class_labels = sorted(os.listdir('Training'))  # must match the training label order

detect_and_display('Testing/notumor/Te-no_103.jpg', model)
```

The function displays the image with the predicted class and confidence.

---

## Important implementation notes

- **Label order:** class labels are derived from `sorted(os.listdir(train_dir))` and reused for inference. A hardcoded list that differs from the training order gives wrong labels even when the model is correct.
- **Preprocessing:** VGG16 expects `preprocess_input`, not simple `/255` scaling. Use the same preprocessing for training and prediction.
- **Evaluation:** use deterministic preprocessing (resize + `preprocess_input` only) on the test set, without random augmentation.

---

## Project structure

```
.
├── Brain_Tumor_Detection.ipynb   # data loading, training, evaluation, inference
├── README.md
```

---

## Possible improvements

- Stronger augmentation (flip, rotation, zoom)
- Two-stage fine-tuning (train head first, then unfreeze `block5` at a low learning rate)
- Validation split with `EarlyStopping` and `ReduceLROnPlateau`
- Class weights to improve glioma recall
- Try EfficientNetB0 or ResNet50V2 backbones
- Grad-CAM visualizations to inspect what the model focuses on

---

## Tech stack

Python, TensorFlow/Keras, NumPy, scikit-learn, Matplotlib, Seaborn


## Acknowledgements

- Dataset: Masoud Nickparvar, Kaggle
- Model: VGG16 (Simonyan and Zisserman, 2014)
