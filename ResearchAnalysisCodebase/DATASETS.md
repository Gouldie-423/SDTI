# Dataset Documentation

This document provides detailed information about the datasets used in the SDTI research project.

---

## Overview

The SDTI framework is evaluated on **17 training datasets** and **11 zero-shot test datasets**, covering both tabular and image data across multiple domains.

---

## Training Datasets (17)

### Tabular Datasets (2)

#### 1. Korean Health Dataset
- **File**: `datasets/smoking_driking_dataset_Ver01.csv`
- **Task**: Binary classification (smoking/drinking prediction)
- **Features**: Health metrics including sex, various health indicators
- **Sample Size**: 3,000 records (sampled from full dataset)
- **Preprocessing**: One-hot encoding for categorical variables (sex, DRK_YN)
- **Target Variable**: `DRK_YN_Y` (drinking status)
- **Source**: Korean health survey data

#### 2. Airline Delay Dataset
- **File**: `datasets/full_data_flightdelay.csv`
- **Task**: Binary classification (flight delay >15 minutes)
- **Features**: Weather conditions, time blocks (excludes high-cardinality columns)
- **Sample Size**: 3,000 records (sampled from full dataset)
- **Removed Columns**: DEP_TIME_BLK, CARRIER_NAME, DEPARTING_AIRPORT, PREVIOUS_AIRPORT
- **Target Variable**: `DEP_DEL15` (delayed indicator)
- **Source**: Flight delay records

---

### Image Datasets - CIFAR-10 (5 binary tasks)

**Base Dataset**: CIFAR-10  
**File**: `datasets/cifar-10-batches-py/`  
**Format**: Pickle files (Python dict)  
**Image Size**: 32×32×3 pixels → flattened to 3,072 features  
**Sample Size**: 60,000 total → 3,000 per binary task  
**Preprocessing**: Flattened, shuffled, normalized

#### Binary Classification Tasks:

1. **CIFAR_Airplane_Automobile**
   - Class 0: Airplane
   - Class 1: Automobile
   - Records: 3,000

2. **CIFAR_Bird_Cat**
   - Class 0: Bird
   - Class 1: Cat
   - Records: 3,000

3. **CIFAR_Deer_Dog**
   - Class 0: Deer
   - Class 1: Dog
   - Records: 3,000

4. **CIFAR_Frog_Horse**
   - Class 0: Frog
   - Class 1: Horse
   - Records: 3,000

5. **CIFAR_Ship_Truck**
   - Class 0: Ship
   - Class 1: Truck
   - Records: 3,000

---

### Image Datasets - Fashion-MNIST (5 binary tasks)

**Base Dataset**: Fashion-MNIST  
**File**: `datasets/fashion-mnist_train.csv`  
**Image Size**: 28×28 grayscale → flattened to 784 features  
**Sample Size**: 60,000 total → 3,000 per binary task  
**Preprocessing**: Flattened, sampled, normalized

#### Binary Classification Tasks:

1. **FASHION_MNIST_Shirt_Trouser**
   - Class 0: T-shirt/top
   - Class 1: Trouser
   - Records: 3,000

2. **FASHION_MNIST_Pullover_Dress**
   - Class 0: Pullover
   - Class 1: Dress
   - Records: 3,000

3. **FASHION_MNIST_Coat_Sandal**
   - Class 0: Coat
   - Class 1: Sandal
   - Records: 3,000

4. **FASHION_MNIST_Shirt_Sneaker**
   - Class 0: Shirt
   - Class 1: Sneaker
   - Records: 3,000

5. **FASHION_MNIST_Bag_Boot**
   - Class 0: Bag
   - Class 1: Ankle boot
   - Records: 3,000

---

### Image Datasets - Digit-MNIST (5 binary tasks)

**Base Dataset**: MNIST Handwritten Digits  
**Files**: `datasets/digit_mnist/MNIST_data.csv` and `MNIST_target.csv`  
**Image Size**: 28×28 grayscale → flattened to 784 features  
**Sample Size**: Full dataset sampled  
**Preprocessing**: Flattened, shuffled, normalized

#### Binary Classification Tasks:

1. **DIGIT_MNIST_01**
   - Class 0: Digit 0
   - Class 1: Digit 1
   - Records: 3,000

2. **DIGIT_MNIST_23**
   - Class 0: Digit 2
   - Class 1: Digit 3
   - Records: 3,000

3. **DIGIT_MNIST_45**
   - Class 0: Digit 4
   - Class 1: Digit 5
   - Records: 3,000

4. **DIGIT_MNIST_67**
   - Class 0: Digit 6
   - Class 1: Digit 7
   - Records: 3,000

5. **DIGIT_MNIST_89**
   - Class 0: Digit 8
   - Class 1: Digit 9
   - Records: 3,000

---

## Zero-Shot Test Datasets (11)

These datasets were used to evaluate SDTI's zero-shot generalization capability.

### SVHN - Street View House Numbers (5 binary tasks)

**File**: `datasets/shvn_train_32x32.mat`  
**Format**: MATLAB .mat file  
**Image Size**: 32×32×3 pixels → flattened to 3,072 features  
**Sample Size**: 73,257 total images  
**Preprocessing**: Reshaped from (32,32,3,N) to (N,3072), normalized

#### Binary Classification Tasks:

1. **SVHN_DATA_01** - Digits 0 vs 1
2. **SVHN_DATA_23** - Digits 2 vs 3
3. **SVHN_DATA_45** - Digits 4 vs 5
4. **SVHN_DATA_67** - Digits 6 vs 7
5. **SVHN_DATA_89** - Digits 8 vs 9

---

### Covertype Dataset (3 binary tasks)

**File**: `datasets/covtype.data.gz`  
**Task**: Forest cover type prediction  
**Features**: Cartographic variables (elevation, slope, soil type, etc.)  
**Sample Size**: Full dataset (581,012 records)  
**Preprocessing**: Compressed gzip file, label shifted by -1, shuffled  
**Original Classes**: 7 forest cover types

#### Binary Classification Tasks:

1. **COVERTYPE_01** - Cover types 1 vs 2
2. **COVERTYPE_23** - Cover types 3 vs 4
3. **COVERTYPE_45** - Cover types 5 vs 6

---

### Crop Mapping Dataset (3 binary tasks)

**File**: `datasets/crop_mapping.txt`  
**Task**: Crop type classification  
**Format**: CSV with comma separator  
**Preprocessing**: Label column separated, converted to binary tasks

#### Binary Classification Tasks:

1. **CROP_MAPPING_01** - Crop types 1 vs 2
2. **CROP_MAPPING_23** - Crop types 3 vs 4
3. **CROP_MAPPING_45** - Crop types 5 vs 6

---

## Data Preprocessing Pipeline

### Binary Classification Conversion

Multi-class datasets are converted to binary classification tasks using the following logic:

```python
def binary_classification_conversion(data, labels):
    # Remove odd-numbered class if total classes is odd
    if len(np.unique(labels)) % 2 != 0:
        data = data[labels != max(labels)]
        labels = labels[labels != max(labels)]
    
    # Create binary tasks from consecutive pairs
    for i in range(1, len(np.unique(labels)), 2):
        filtered_data = data[(labels == i-1) | (labels == i)][:3000]
        filtered_labels = np.where(labels == i, 1, 0)[:3000]
        export_data.append([filtered_data, filtered_labels])
    
    return export_data
```

**Rationale for Binary Conversion:**
1. More rigorous evaluation when datasets and labels originate from the same domain
2. Increases task difficulty (1-in-2 chance vs 1-in-N for multi-class)
3. Higher risk of incorrect associations in same-domain data

---

### Normalization

Each dataset is independently normalized using **z-score normalization**:

```python
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data[:2000])
```

**Key Points:**
- Applied independently per dataset (no shared normalization)
- Computed only from training subset (first 2,000 records)
- Prevents data leakage

---

### Zero-Padding

Datasets have different feature dimensions and are zero-padded to match the maximum:

```python
max_size = max([dataset.shape[1] for dataset in all_datasets])
for dataset in datasets:
    if dataset.shape[1] < max_size:
        padding_size = max_size - dataset.shape[1]
        dataset = np.concatenate([dataset, np.zeros((m, padding_size))], axis=1)
```

**Purpose:** Enable vectorized processing in 3D tensors

---

## Dataset Statistics

### Feature Dimensions

| Dataset Type | Original Dimensions | Padded Dimensions |
|-------------|---------------------|-------------------|
| CIFAR-10 | 3,072 | 3,072 (max) |
| Fashion-MNIST | 784 | 3,072 |
| Digit-MNIST | 784 | 3,072 |
| Korean Health | ~20 | 3,072 |
| Airline Delay | ~10 | 3,072 |
| SVHN | 3,072 | 3,072 |
| Covertype | 54 | 3,072 |
| Crop Mapping | Variable | 3,072 |

### Memory Requirements

- **Training Datasets Tensor**: (289, 2000, 3072) = 289 dataset-label combinations × 2000 records × 3072 features
- **Labels Tensor**: (289, 2000, 1)
- **Total Size**: ~6.64 GB (including weights, biases, Adam optimizer states)

---

## Dataset Sources

### Where to Download

1. **CIFAR-10**: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
2. **Fashion-MNIST**: [https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)
3. **Digit-MNIST**: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
4. **SVHN**: [http://ufldl.stanford.edu/housenumbers/](http://ufldl.stanford.edu/housenumbers/)
5. **Covertype**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/covertype)
6. **Korean Health**: Available on Kaggle
7. **Airline Delay**: Available on Kaggle
8. **Crop Mapping**: Contact authors or check supplementary materials

---

## Data Loading Functions

All data loading is handled by `dataset_preprocessing.py`:

### Main Loading Functions

```python
# Load training datasets
datasets = load_datasets()  # Returns dict with 17 datasets

# Load zero-shot test datasets
test_datasets = load_zero_shot_datasets()  # Returns dict with 11 datasets

# Normalize and create metadata
train_datasets, ground_truth_labels = normalize_and_create_metadata(datasets)

# Zero-pad to uniform dimensions
train_datasets = zero_pad_datasets(train_datasets)

# Generate data structure for SDTI model
train_data, train_labels = data_structure_gen(train_datasets)
```

### Individual Dataset Functions

- `cifar10_preprocessing(path)` - Load and convert CIFAR-10
- `fashion_mnist_preprocessing(path)` - Load and convert Fashion-MNIST
- `digit_mnist_preprocessing(path)` - Load and convert Digit-MNIST
- `airline_data_preprocessing(path)` - Load airline delay data
- `korean_health_preprocessing(path)` - Load Korean health data
- `shvn_preprocessing(path)` - Load SVHN data
- `covertype_preprocessing(path)` - Load Covertype data
- `crop_mapping_preprocessing(path)` - Load crop mapping data

---

## Notes on Dataset Selection

### Why These Datasets?

1. **Diversity**: Cover multiple domains (images, tabular) and tasks
2. **Benchmark Status**: Well-known datasets commonly used in ML research
3. **Accessibility**: Publicly available from standard repositories
4. **Complexity**: Semantic relationships in flattened images are subtle and difficult to detect
5. **Binary Tasks**: Enable stress-testing of SDTI's discriminative ability

### Flattened Image Data

Images are flattened rather than processed with CNNs because:
1. Semantic relationships become more subtle and challenging
2. Tests SDTI's ability to identify patterns in high-dimensional, seemingly unstructured data
3. Consistent with the single-neuron ANN architecture
4. Demonstrates that manifold complexity signal works even without spatial structure

---

## Reproducibility

**Random Seed**: Set to 42 in preprocessing functions for reproducible sampling

```python
df = df.sample(n=3000, random_state=42)
```

This ensures identical train/test splits across runs.

---

*For questions about datasets, please refer to `dataset_preprocessing.py` or contact the authors.*
