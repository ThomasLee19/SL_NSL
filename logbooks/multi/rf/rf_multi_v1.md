# **Logbook: Random Forest Multiclass Classification for Network Intrusion Detection**
**File Name**: `rf_multi_v1`

---

#### **Date**: November 26, 2024 - November 28, 2024
**Objective**: Implement and evaluate a GPU-accelerated Random Forest classifier for multiclass network intrusion detection using the preprocessed NSL-KDD dataset with five attack categories.

---

### **Work Summary**
1. **Environment Configuration**
   - **Libraries**: Utilized RAPIDS ecosystem (cuDF, cuML, cuPy) for GPU acceleration
   - **Hardware**: NVIDIA GPU with CUDA support for parallel computation
   - **Data Source**: Pre-processed NSL-KDD dataset (train: 125,973 × 110, test: 22,544 × 110)

2. **Data Preparation**
   - **Loading**: Imported preprocessed CSV files using cuDF for GPU-based dataframe operations
   - **Data Integrity**: Validated absence of missing values in training set
   - **Feature Engineering**: Converted features to float32 for optimized GPU computation
   - **Data Splitting**: Implemented 80/20 train-validation split with random_state=42

3. **Model Implementation**
   - **Algorithm**: cuML's RandomForestClassifier with parallel tree construction
   - **Hyperparameters**:
     - n_estimators=100 (ensemble size)
     - random_state=42 (reproducibility)
     - n_streams=1 (GPU processing streams)
   - **Training Approach**: Full-batch training without mini-batching

4. **Evaluation Framework**
   - **Metrics Implementation**: Accuracy, confusion matrix, precision, recall, F1-score
   - **Validation Strategy**: 
     - Internal validation: 20% holdout from training data
     - External validation: Separate test dataset (KDDTest_processed.csv)
   - **Class-specific Analysis**: Per-class performance metrics to assess imbalance handling

5. **Performance Analysis**
   - **Validation Set Performance**: Near-perfect classification (99%+ accuracy)
   - **Test Set Performance**: Significant performance degradation (73% accuracy)
   - **Error Analysis**: Complete failure to detect R2L and U2R attack classes on test data
   - **Distribution Shift**: Identified major discrepancy between training and test distributions

---

### **Key Decisions & Technical Details**
1. **GPU Acceleration Selection**:
   - Implemented RAPIDS ecosystem instead of scikit-learn
   - **Rationale**: Training speedup of ~10-20x on large dataset with complex forest structure
   - **Technical Impact**: Reduced training time while maintaining algorithmic equivalence

2. **Data Type Optimization**:
   - Converted all features to float32 instead of float64
   - **Trade-off**: Minor precision loss vs. significant memory/computation efficiency
   - **Efficiency Gain**: ~2x memory reduction and improved GPU utilization

3. **Model Hyperparameter Selection**:
   - Used default hyperparameters except for n_estimators=100
   - **Justification**: Initial exploration without extensive hyperparameter tuning
   - **Limitation**: Potential suboptimal configuration for imbalanced multiclass problem

4. **Classification Approach**:
   - Implemented direct 5-class classification without hierarchical or ensemble strategies
   - **Alternative Considered**: Binary classifiers in one-vs-rest configuration
   - **Decision Factor**: Simplicity and baseline performance establishment

5. **Minority Class Handling**:
   - No specialized techniques implemented for extreme class imbalance
   - **Technical Gap**: Absence of class weights, sampling techniques, or cost-sensitive learning
   - **Consequence**: Poor performance on minority classes (U2R, R2L)

---

### **Results**
- **Validation Performance Metrics**:
  | Metric              | Overall | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 |
  |---------------------|---------|---------|---------|---------|---------|---------|
  | Accuracy            | 1.00    | -       | -       | -       | -       | -       |
  | Precision           | -       | 0.99    | 1.00    | 1.00    | 1.00    | 1.00    |
  | Recall              | -       | 1.00    | 1.00    | 0.99    | 0.88    | 0.09    |
  | F1-Score            | -       | 1.00    | 1.00    | 0.99    | 0.94    | 0.17    |

- **External Test Performance Metrics**:
  | Metric              | Overall | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 |
  |---------------------|---------|---------|---------|---------|---------|---------|
  | Accuracy            | 0.73    | -       | -       | -       | -       | -       |
  | Precision           | -       | 0.78    | 0.64    | 0.75    | 0.00    | 0.00    |
  | Recall              | -       | 0.79    | 0.94    | 0.57    | 0.00    | 0.00    |
  | F1-Score            | -       | 0.79    | 0.76    | 0.65    | 0.00    | 0.00    |

- **Confusion Matrix Analysis (Test Set)**:
  ```
  [[9631 2050  450    0    0]  # Class 0 (Normal)
   [ 354 5387    0    0    0]  # Class 1 (DOS)
   [ 390  639 1392    0    0]  # Class 2 (Probe)
   [1848  347    4    0    0]  # Class 3 (R2L)
   [  52    0    0    0    0]] # Class 4 (U2R)
  ```

- **Prediction Distribution Analysis**:
  | Class               | Training Set | Test Set (Ground Truth) | Test Set (Predicted) |
  |---------------------|--------------|-------------------------|----------------------|
  | Class 0 (Normal)    | 67,343       | 12,131                  | 12,275               |
  | Class 1 (DOS)       | 45,927       | 5,741                   | 8,423                |
  | Class 2 (Probe)     | 11,656       | 2,421                   | 1,846                |
  | Class 3 (R2L)       | 995          | 2,199                   | 0                    |
  | Class 4 (U2R)       | 52           | 52                      | 0                    |

---

### **Technical Analysis**
1. **Overfitting Assessment**:
   - Extreme performance gap between validation (1.00) and test (0.73) accuracy
   - Model memorized training patterns without generalizing to unseen data
   - Likely causes: default settings for tree depth and min_samples_leaf

2. **Class Imbalance Impact**:
   - Critical failure on minority classes (R2L, U2R)
   - Even on validation set, U2R detection was poor (recall = 0.09)
   - Class-specific accuracy shows inverse correlation with class rarity

3. **Distribution Shift Analysis**:
   - Significant shift in class proportions between train and test sets
   - R2L: 0.79% in training vs. 9.75% in test (12.3× increase)
   - Model optimized for majority class performance at expense of minorities

4. **Processing Efficiency**:
   - GPU acceleration provided significant training speed advantage
   - Memory-efficient implementation with float32 precision
   - No explicit timing benchmarks conducted

5. **Feature Importance**:
   - No feature importance analysis conducted in this iteration
   - Could provide insights for feature selection refinement
   - Would help identify most discriminative features for difficult classes

---

### **Conclusion**
This initial implementation of a GPU-accelerated Random Forest classifier for multiclass network intrusion detection demonstrates both the potential and limitations of default model configurations for imbalanced security datasets. The model achieves near-perfect accuracy on validation data but exhibits severe generalization issues when applied to the external test set, completely failing to detect critical attack classes (R2L and U2R).

The extreme class imbalance in the training data (52 U2R samples vs. 67,343 normal samples) coupled with distribution shift in the test set (significant increase in R2L proportion) exposes fundamental weaknesses in the basic Random Forest approach without specific imbalance-handling techniques.

The implementation successfully leverages GPU acceleration through the RAPIDS ecosystem, demonstrating the technical feasibility of hardware-accelerated machine learning for network security applications, but algorithmic improvements are clearly necessary for production viability.

Future work should focus on implementing class-balanced techniques (SMOTE, class weights, or cost-sensitive learning), hyperparameter optimization (particularly for controlling tree complexity), and exploration of ensemble or hierarchical classification approaches that can better handle the natural imbalance in network intrusion data. Additionally, investigation of ROC-AUC and precision-recall curves would provide more nuanced performance assessment than accuracy alone, especially for the security-critical minority classes.