# Self-Directed Task Identification (SDTI)

This repository contains the implementation and experimental code for the research paper:

**"Self-Directed Task Identification"**  
*Timothy Gould and Sidike Paheding*  
*Fairfield University, Fairfield, CT, USA*

arXiv: [2604.02430v1](https://arxiv.org/abs/2604.02430v1) [cs.LG] 2 Apr 2026

---

## Abstract

Self-Directed Task Identification (SDTI) is a novel machine learning framework that enables models to autonomously identify the correct target variable for each dataset in a zero-shot setting without pre-training. SDTI demonstrates the feasibility of repurposing core machine learning concepts for a novel task structure using only standard neural network components.

Traditional approaches lack the capability to autonomously determine correct target variables, leaving data annotation as a time-consuming process that relies heavily on human effort. This work shows that SDTI can be achieved through appropriate problem formulation and architectural design.

**Key Results:**
- **14% improvement** in F1 score over baseline methods (Pearson correlation, Mutual Information, Cosine Similarity)
- **99% F1 score** achieved with optimal hyperparameters (≥500 records, ≥10 epochs, 30 SDTI epochs)
- Demonstrated effectiveness across 17 benchmark datasets in zero-shot learning settings

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Usage](#usage)
- [Experimental Results](#experimental-results)
- [Model Architecture](#model-architecture)
- [Citation](#citation)
- [License](#license)

---

## Overview

### What is SDTI?

SDTI refers to a model's capacity to infer the correct dataset-target variable pairings when presented with multiple datasets and their associated labels, without prior knowledge of which dataset corresponds to which target variable.

### How Does It Work?

The SDTI model leverages **manifold complexity** as an implicit supervisory signal through a novel vectorized architecture:

1. **3D Tensor Architecture**: All datasets are concatenated into a single 3-dimensional tensor (N²_D, m, c) where:
   - N_D = number of datasets in the corpus
   - N²_D = number of dataset-label combinations (N_D × N_D)
   - m = number of records per dataset
   - c = maximum feature dimension (datasets zero-padded to this size)
   
2. **Simultaneous Training**: Every dataset-label combination trains its own single-neuron ANN in parallel through vectorized operations - all N²_D ANNs learn simultaneously in one forward/backward pass

3. **Cost-Based Selection**: After training, the model identifies correct pairings by selecting combinations with the lowest cost (argmax across the cost matrix)

4. **Manifold Resistance**: Incorrect dataset-label pairings yield more complex, harder-to-optimize manifolds with higher costs, creating a discriminative signal

5. **Zero-Shot Learning**: Operates without pre-training or external supervision - the optimization difficulty itself provides the supervisory signal

**This Experiment**: N_D = 17 datasets, m = 2000 records, c = 3072 features → Tensor shape (289, 2000, 3072)

### Core Innovation

When data and target variables are mismatched, their semantic relationship is corrupted, distorting the underlying function and yielding a manifold that is more chaotic and difficult to optimize. SDTI exploits this property to distinguish correct from incorrect pairings.

---

## Installation

### Requirements

```bash
python >= 3.8
torch >= 1.9.0
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 0.24.0
scipy >= 1.7.0
matplotlib >= 3.4.0
torchmetrics >= 0.5.0
boto3 >= 1.18.0  # For S3 integration (optional)
PIL >= 8.3.0
```

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd ResearchAnalysisCodebase

# Install dependencies
pip install torch numpy pandas scikit-learn scipy matplotlib torchmetrics pillow

# Optional: For AWS S3 integration
pip install boto3
```

### Dataset Preparation

Place the following datasets in the `datasets/` directory:

- `full_data_flightdelay.csv` - Airline delay data
- `smoking_driking_dataset_Ver01.csv` - Korean health data
- `cifar-10-batches-py/` - CIFAR-10 image dataset
- `fashion-mnist_train.csv` - Fashion-MNIST dataset
- `digit_mnist/` - Digit MNIST dataset
- `covtype.data.gz` - Covertype dataset
- `shvn_train_32x32.mat` - SVHN dataset
- `crop_mapping.txt` - Crop mapping dataset

**Note:** Most datasets can be downloaded from standard ML repositories (UCI ML Repository, Kaggle, official dataset websites).

---

## Repository Structure

```
ResearchAnalysisCodebase/
│
├── README.md                          # This file
├── dataset_preprocessing.py           # Data loading and preprocessing utilities
│
├── SDTI Pipeline.ipynb                # Main SDTI model implementation
├── BaselineTests.ipynb                # Baseline comparison tests
├── Results Analysis.ipynb             # Ablation study analysis
├── CostDataAnalysis.ipynb             # Cost trajectory analysis
│
├── TestResults.csv                    # Ablation study results (3,600 configurations)
├── CostResults.csv                    # Detailed cost trajectories
├── correct_cost_vals.npy              # Cost values for correct pairings
├── incorrect_cost_vals.npy            # Cost values for incorrect pairings
│
├── MeanValues.jpg                     # Cost trajectory visualization
└── ablation_results_enhanced.jpg      # 3D ablation study visualization
```

### File Descriptions

#### Core Files

- **`dataset_preprocessing.py`**: Contains all data loading, preprocessing, normalization, and binary classification conversion functions. **DO NOT MODIFY** - used for reproducibility.

- **`SDTI Pipeline.ipynb`**: Main notebook containing:
  - `SDTIModel` class - Main model wrapper
  - `SDTILayer` class - Vectorized single-neuron ANN implementation
  - Training loop with nested SDTI epochs
  - Hyperparameter generation (learning rates, beta values)
  - Cost computation and prediction logic

#### Analysis Files

- **`BaselineTests.ipynb`**: Implements baseline comparison methods:
  - Pearson's correlation
  - Mutual Information
  - Cosine similarity
  
- **`Results Analysis.ipynb`**: Comprehensive ablation study analysis:
  - 3,600 hyperparameter combinations tested
  - Parameter sensitivity analysis (records, epochs, SDTI epochs)
  - Statistical comparisons and visualizations

- **`CostDataAnalysis.ipynb`**: Cost trajectory analysis:
  - Mean and variance of cost values for correct vs. incorrect pairings
  - Levene's test for variance equality
  - Manifold complexity visualization

#### Results Files

- **`TestResults.csv`**: Contains F1 scores, accuracy, memory usage, and time-to-completion for all 3,600 ablation configurations
- **`CostResults.csv`**: Detailed cost values at each SDTI epoch for all dataset-label combinations
- **`*.npy` files**: Binary numpy arrays for efficient cost data storage
- **`*.jpg` files**: Publication-quality visualizations

---

## Datasets

### Training Datasets (17 total)

The model is evaluated on 17 benchmark datasets:

**Tabular Data (2 datasets):**
1. Korean Health (Smoking/Drinking prediction)
2. Airline Delay Data (Flight delay prediction)

**Image Data - Binary Classification (15 datasets):**
- CIFAR-10 (5 binary tasks): Airplane/Automobile, Bird/Cat, Deer/Dog, Frog/Horse, Ship/Truck
- Fashion-MNIST (5 binary tasks): Shirt/Trouser, Pullover/Dress, Coat/Sandal, Shirt/Sneaker, Bag/Boot
- Digit-MNIST (5 binary tasks): 0/1, 2/3, 4/5, 6/7, 8/9

### Zero-Shot Test Datasets (11 datasets)

For zero-shot evaluation:
- SVHN (Street View House Numbers) - 5 binary tasks
- Covertype - 3 binary tasks  
- Crop Mapping - 3 binary tasks

### Preprocessing Details

- **Normalization**: Z-score normalization applied independently per dataset
- **Sample Size**: Maximum 2,000 samples per dataset (configurable)
- **Binary Conversion**: Multi-class datasets partitioned into binary classification tasks
- **Zero-Padding**: All datasets padded to maximum feature dimension (3,072) - **this is critical for vectorization**
- **Train/Test Split**: First 2,000 records used for training

### Why Zero-Padding?

Zero-padding all datasets to the same feature dimension enables **3D tensor operations**:

```python
# Without padding: Cannot stack datasets with different feature sizes
Dataset 1: (m, f₁)   # f₁ features
Dataset 2: (m, f₂)   # f₂ features (f₂ ≠ f₁)
# These cannot be stacked into a 3D tensor!

# With padding: All datasets have uniform shape
Dataset 1: (m, c)    # Padded with (c - f₁) zeros
Dataset 2: (m, c)    # Padded with (c - f₂) zeros
# Now they stack into: (N²_D, m, c) for simultaneous processing

# Example from this experiment:
# Fashion-MNIST: (2000, 784) → (2000, 3072) padded with 2288 zeros
# CIFAR-10:      (2000, 3072) → (2000, 3072) no padding needed
# Stacked:       (289, 2000, 3072)
```

This allows all N²_D dataset-label combinations to be tested **simultaneously** in a single vectorized operation, reducing training time from O(N²_D) sequential iterations to O(1) parallel operation - a **100× speedup** in practice.

---

## Usage

### 1. Running the Full SDTI Pipeline

```python
# Open SDTI Pipeline.ipynb in Jupyter
jupyter notebook "SDTI Pipeline.ipynb"

# The notebook will:
# 1. Load and preprocess all 17 datasets
# 2. Create the SDTI model with specified hyperparameters
# 3. Train across multiple epochs with nested SDTI layer iterations
# 4. Save results to TestResults.csv and CostResults.csv
# 5. Upload results to S3 (if configured)
```

### 2. Running Baseline Comparisons

```python
# Open BaselineTests.ipynb
jupyter notebook "BaselineTests.ipynb"

# Implements three baseline methods:
# - Pearson's Correlation
# - Mutual Information
# - Cosine Similarity
```

### 3. Analyzing Results

```python
# Open Results Analysis.ipynb
jupyter notebook "Results Analysis.ipynb"

# Performs comprehensive analysis:
# - Load ablation study results
# - Generate 3D visualization
# - Compare performance across parameter settings
# - Compute statistical summaries
```

### 4. Analyzing Cost Trajectories

```python
# Open CostDataAnalysis.ipynb
jupyter notebook "CostDataAnalysis.ipynb"

# Analyzes cost evolution:
# - Mean/variance of correct vs incorrect pairings
# - Statistical significance testing (Levene's test)
# - Manifold complexity visualization
```

### Key Hyperparameters

The ablation study tested 3,600 combinations of:

```python
# Number of training records
A = {2000, 1900, 1800, ..., 200, 100}  # 20 values

# Number of model training epochs  
B = {1, 3, 5, 7, ..., 57, 59}  # 30 values (odd numbers only)

# Number of SDTI layer iterations
C = {30, 25, 20, 15, 10, 5}  # 6 values
```

**Optimal Configuration:**
- Records: ≥ 500
- Epochs: ≥ 10  
- SDTI Epochs: 30
- **Result: 99% mean F1 score**

---

## Experimental Results

### Performance Summary

| Configuration | Mean F1 | Std Dev | Min F1 | 25th % | 50th % | 75th % | Max F1 |
|--------------|---------|---------|--------|--------|--------|--------|--------|
| **Improved Parameters** | 0.9900 | 0.0262 | 0.9216 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Worse Parameters | 0.7193 | 0.1712 | 0.3529 | 0.6358 | 0.7647 | 0.8431 | 0.9216 |

*Improved Parameters: records ≥ 500, epochs ≥ 10, SDTI epochs = 30 (400 runs)*  
*Worse Parameters: records < 500, epochs < 10, SDTI epochs = 5 (20 runs)*

### Comparison with Baselines

| Algorithm | Mean F1 | Standard Deviation |
|-----------|---------|-------------------|
| **SDTI** | **0.9900** | 0.0262 |
| Pearson's Correlation | 0.8431 | 0.0000 |
| Mutual Information | 0.8431 | 0.0000 |
| Cosine Similarity | 0.8431 | 0.0000 |

**SDTI achieves 14.69% improvement** over all baseline algorithms.

### Manifold Complexity Analysis

| Group | Mean Cost (μ) | Variance (σ²) |
|-------|--------------|---------------|
| Correct Pairs | 0.6483 | 0.2597 |
| Incorrect Pairs | 0.9913 | 0.6093 |

**Statistical Significance:**
- Levene's test: p = 8.77 × 10⁻¹⁹⁵
- Conclusion: Incorrect pairings show **significantly higher cost variance**, confirming more complex manifolds

### Computational Performance

**Training Time:**
- Vectorized implementation: ~3 seconds per iteration
- Non-vectorized: ~5 minutes per iteration (100× slower)
- Full ablation study: Average 98 seconds (std: 55 seconds) per run

**Memory Usage:**
- Scales as O(N²_D · m · c) where N_D = datasets, m = records, c = features
- This experiment: N_D=17 (289 combinations), m=2000, c=3072 → ~6.64 GB GPU memory

**Hardware:**
- AWS SageMaker ml.g4dn.xlarge
- NVIDIA T4 Tensor Core GPU (16 GB)
- 4 vCPUs, 16 GB system memory

---

## Model Architecture

### Algorithm Overview

```
Algorithm: SDTI Training and Prediction Workflow

For epoch = 1 → E (main training epochs):
    1. Sample learning rates and beta values for each dataset
    2. Initialize weights and biases in SDTI Layer
    
    For sdti_epoch = 1 → S (nested SDTI epochs):
        3. Forward propagation for all dataset/label combinations (vectorized)
        4. Compute cost for each combination (289 costs simultaneously)
        5. Compute gradients (vectorized backpropagation)
        6. Update weights using Adam optimizer (all 289 ANNs updated in parallel)
    
    7. Reshape costs to (N×N) matrix and record prediction (argmax per row)

8. Aggregate predictions across all epochs (mode)
9. Return final predictions
```

### SDTI Layer Architecture - The Key Innovation

The SDTI Layer is the core innovation enabling zero-shot task identification:

1. **3D Tensor Structure**: 
   - **Data**: (N²_D, m, c) tensor containing all dataset-label combinations
   - **Weights**: (N²_D, c, 1) tensor - unique weights for each combination
   - **Biases**: (N²_D, 1, 1) tensor - unique bias for each combination

2. **Vectorized Single-Neuron ANNs**: Each of the N²_D ANNs is just a single neuron (weighted sum + sigmoid), but they all execute **simultaneously** through 3D matrix operations

3. **Independent Learning**: Despite sharing the same tensor structure, each ANN:
   - Has unique randomly sampled hyperparameters (learning rate, betas)
   - Learns independently from its specific data-label pair
   - Optimizes its own cost function
   - Never shares gradients with other ANNs

4. **Parallel Processing**: All N²_D forward passes, cost calculations, gradient computations, and weight updates happen in a **single vectorized operation**

5. **Cost-Based Selection**: After training, the model outputs a (N²_D,) cost vector that is reshaped to (N_D, N_D), where cost[i,j] represents how well dataset i fits with label set j. The argmax along each row identifies the predicted label for each dataset.

**Key Equations:**

Forward Propagation (Vectorized):
```
Ô_N²_D×m×1 = σ((A_N²_D×m×c · W_N²_D×c×1) + B_N²_D×1×1)
```
This single operation computes outputs for all N²_D dataset-label combinations simultaneously.

Cost Function (Binary Cross-Entropy, Vectorized):
```
Loss_N²_D×m×1 = -[y·log(Ô + ε) + (1-y)·log(1-Ô + ε)]
Cost_N²_D×1 = mean(Loss, axis=1)  # Average over m records
```
Produces N²_D cost values in one operation.

Prediction (Per Epoch):
```
Cost_matrix_N_D×N_D = reshape(Cost_N²_D×1, (N_D, N_D))
ŷ_N_D×1 = argmax(Cost_matrix, axis=1)  # Best label for each dataset
```

**The Power of Vectorization:**
- Sequential: N²_D iterations × t seconds = O(N²_D · t) time per SDTI epoch
- Vectorized: 1 operation × t seconds = O(t) time per SDTI epoch
- **Speedup: O(N²_D)** — scales quadratically with number of datasets
- **This experiment (N_D=17)**: 289 iterations → 1 operation = ~100× faster

### 3D Tensor Data Structure Explained

This is the key architectural innovation that enables SDTI to scale:

**Step 1: Zero-Pad Individual Datasets**
```python
# Each dataset D_i has different feature dimensions f_i:
D_1: (m, f_1)
D_2: (m, f_2)
...
D_N_D: (m, f_N_D)

# After zero-padding to max dimension c = max(f_i):
D_1: (m, c)  # Added (c - f_1) zero columns
D_2: (m, c)  # Added (c - f_2) zero columns
...
D_N_D: (m, c)  # Added (c - f_N_D) zero columns

# Example from this experiment (N_D=17, m=2000, c=3072):
# CIFAR-10: (2000, 3072) → (2000, 3072)  # No padding needed
# Fashion-MNIST: (2000, 784) → (2000, 3072)  # Added 2288 zeros
# Korean Health: (2000, ~20) → (2000, 3072)  # Added ~3052 zeros
```

**Step 2: Create All Dataset-Label Combinations**
```python
# For N_D datasets and N_D label sets:
# Test dataset_0 with all N_D label sets → N_D combinations
# Test dataset_1 with all N_D label sets → N_D combinations
# ...
# Total: N_D × N_D = N²_D combinations

# Stack into 3D tensor:
Data_tensor: (N²_D, m, c)
Label_tensor: (N²_D, m, 1)

# Example: (289, 2000, 3072) and (289, 2000, 1)
```

**Step 3: Vectorized Training**
```python
# Each of the N²_D combinations gets its own:
Weights: (N²_D, c, 1)  # Unique weights per combination
Biases: (N²_D, 1, 1)   # Unique bias per combination

# Single matrix multiplication trains all N²_D ANNs:
Output = sigmoid(Data_tensor @ Weights + Biases)
# Result: (N²_D, m, 1) - predictions for all combinations

# Cost computation for all combinations:
Costs = mean(BCE_loss(Output, Label_tensor), axis=1)
# Result: (N²_D,) - one cost per combination

# Example: (289,) cost vector
```

**Step 4: Identify Correct Pairings**
```python
# Reshape costs to matrix form:
Cost_matrix = reshape(Costs, (N_D, N_D))
# Cost_matrix[i,j] = how well dataset_i fits with label_j

# Find best label for each dataset:
Predictions = argmax(Cost_matrix, axis=1)
# Predictions[i] = predicted label index for dataset_i

# Example: reshape (289,) → (17, 17), then argmax per row
```

### Hyperparameter Generation

Learning rates sampled from log-uniform distribution:
```
lr_i ~ 10^u_i, where u_i ~ Uniform(-5, -1)
```

Adam beta parameters:
```
β₁ ~ Uniform(0.85, 0.99)
β₂ ~ Uniform(0.98, 0.9999)
```

Hyperparameters are broadcast to match 3D tensor dimensions:
```python
# Sample once per dataset (N_D values):
lrs = [lr_0, lr_1, ..., lr_(N_D-1)]

# Broadcast to all combinations (N²_D values):
lrs_broadcast = [lr_0, lr_0, ..., lr_0,      # dataset_0 with N_D labels
                 lr_1, lr_1, ..., lr_1,      # dataset_1 with N_D labels
                 ...
                 lr_(N_D-1), ..., lr_(N_D-1)]  # dataset_(N_D-1) with N_D labels

# Example (N_D=17): 17 sampled → 289 broadcast values
```

---

## Key Insights

### Why SDTI Works

1. **Manifold Complexity as Signal**: Incorrect dataset-label pairings create more complex, irregular manifolds that resist optimization

2. **Cost Variance**: Incorrect pairings show 2.35× higher cost variance (0.6093 vs 0.2597)

3. **No Pretraining Required**: SDTI operates in a true zero-shot setting without external supervision

4. **Scalable Architecture**: Vectorization enables parallel processing of all combinations

### Limitations

1. **Scalability**: Memory grows as O(N²_D · m · c) - quadratically with number of datasets. For example:
   - N_D = 17 → 289 combinations → ~6.6 GB
   - N_D = 30 → 900 combinations → ~20 GB
   - A batching framework is needed for larger corpora
   
2. **Binary Classification Only**: Current implementation limited to binary tasks

3. **Class Imbalance Sensitivity**: Highly imbalanced datasets may cause weight collapse

4. **No Unknown Token**: Cannot detect when correct label is not in the corpus

5. **Tabular Data Only**: Does not yet support multimodal inputs

---

## Future Work

Planned extensions include:

1. **Multimodal Support**: Extend to vision, text, and audio modalities
2. **Imbalanced Data Handling**: Improved loss functions for class imbalance
3. **Batching Framework**: Memory-efficient processing for larger dataset collections
4. **Unknown Token**: Detection mechanism when correct label is absent
5. **Multi-class Support**: Direct multi-class classification without binary conversion
6. **Production Deployment**: Integration into automated data annotation pipelines

---

## Citation

If you use this code or find this research helpful, please cite:

```bibtex
@article{gould2026sdti,
  title={Self-Directed Task Identification},
  author={Gould, Timothy and Paheding, Sidike},
  journal={arXiv preprint arXiv:2604.02430},
  year={2026},
  institution={Fairfield University}
}
```

---

## License

This code is provided for research purposes. Please contact the authors for commercial use.

---

## Contact

**Authors:**
- Timothy Gould - Fairfield University
- Sidike Paheding - Fairfield University

**Institution:**  
Fairfield University  
Fairfield, CT, USA

**Paper:** [arXiv:2604.02430](https://arxiv.org/abs/2604.02430)

---

## Acknowledgments

This research was conducted at Fairfield University. Experiments were performed on AWS SageMaker infrastructure.

---

## Reproducibility

To reproduce the results in the paper:

1. Download all required datasets to `datasets/` directory
2. Run `SDTI Pipeline.ipynb` with the ablation study configuration (3,600 runs)
3. Analyze results using `Results Analysis.ipynb` and `CostDataAnalysis.ipynb`
4. Compare with baselines using `BaselineTests.ipynb`

**Expected Runtime:** ~100 hours for full ablation study (3,600 configurations)

**Note:** Random seed is set to 42 in preprocessing for reproducibility of dataset sampling.

---

*Last Updated: October 2026*
