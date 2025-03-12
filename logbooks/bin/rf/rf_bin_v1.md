# **Logbook: Supervised Binary Classification with Random Forest Version 1.0**
**File Name**: `rf_bin_v1`

---

#### **Date**: November 1, 2024 - November 4, 2024
**Objective**: Train and evaluate a supervised Random Forest Classifier for binary classification on the preprocessed NSL-KDD dataset, comparing validation performance with external test set results to assess model generalization capabilities.

---

### **Work Summary**
1. **Data Loading & Preparation**
   - **Training Data**: Loaded preprocessed `KDDTrain_processed.csv` (125,973 samples, 108 features).
   - **Testing Data**: Loaded preprocessed `KDDTest_processed.csv` (22,544 samples, 108 features).
   - **Class Distribution**: 
     - Training data: 67,343 normal (53.5%), 58,630 attack (46.5%)
   - **Key Actions**:
     - Split data into train (80%, 100,778 samples) and validation (20%, 25,195 samples) sets, preserving class distribution.
     - Converted all features to float32 data type for GPU compatibility.

2. **Model Configuration & Training**
   - Implemented Random Forest Classifier with GPU acceleration via CUDF and CuML.
   - **Hyperparameters**:
     - `n_estimators`: 50 trees (balancing performance and efficiency)
     - `random_state`: 42 (for reproducibility)
     - `n_streams`: 1 (for GPU processing)

3. **Validation Evaluation**
   - Assessed model performance on 20% validation split:
     - **Metrics**: Accuracy, precision, recall, F1-score
     - **Confusion Matrix Analysis**: Evaluated false positive and false negative rates
   - **Results**: Near-perfect classification on validation data

4. **External Test Set Assessment**
   - Evaluated model generalization on separate test dataset:
     - Applied identical preprocessing steps for consistency
     - Compared performance metrics with validation results
     - Analyzed error patterns through confusion matrix

---

### **Results**
- **Model Performance**: 
  | Dataset    | Accuracy | Precision (Avg) | Recall (Avg) | F1-Score (Avg) |
  |------------|----------|-----------------|--------------|----------------|
  | Validation | 1.00     | 1.00            | 1.00         | 1.00           |
  | Test       | 0.81     | 0.82            | 0.81         | 0.81           |

- **Validation Confusion Matrix**:
  - True Negatives: 13,409 (correctly identified normal)
  - False Positives: 13 (normal misclassified as attack)
  - False Negatives: 73 (attack misclassified as normal)
  - True Positives: 11,700 (correctly identified attack)

- **Test Confusion Matrix**:
  - True Negatives: 8,369 (correctly identified normal)
  - False Positives: 1,342 (normal misclassified as attack)
  - False Negatives: 2,916 (attack misclassified as normal)
  - True Positives: 9,917 (correctly identified attack)

---

### **Key Decisions**
1. **Model Selection**:
   - Chose supervised Random Forest over unsupervised methods.
   - **Reasoning**: Leveraging available labels for direct optimization of classification performance.

2. **GPU Acceleration**:
   - Implemented RAPIDS ecosystem (cuDF, cuML) for GPU-accelerated training and inference.
   - **Trade-off**: Increased setup complexity for significantly reduced training time.

3. **Ensemble Size**:
   - Selected 50 estimators for Random Forest.
   - **Rationale**: Preliminary testing showed minimal performance gains beyond 50 trees while increasing computational cost.

4. **Data Type Optimization**:
   - Converted features to float32 instead of default float64.
   - **Benefit**: Reduced memory usage and improved GPU processing efficiency with minimal precision loss.

---

### **Conclusion**
The Random Forest model (rf_bin_v1) demonstrated exceptional performance on the validation set with near-perfect metrics across accuracy, precision, recall, and F1-score. However, there was a noticeable performance drop on the external test set (accuracy: 0.81), indicating potential overfitting or distribution shift between training and test data.

The model showed good balance between precision and recall on the test set, with an overall F1-score of 0.81, significantly outperforming the unsupervised Isolation Forest approach (F1-score: 0.715) documented in if_bin_v1. The supervised nature of Random Forest provided clear advantages in classification performance.

The performance gap between validation and test results warrants further investigation into potential dataset drift, feature importance analysis, or hyperparameter optimization to improve generalization. Future work should explore methods to reduce false negatives (2,916) on the test set, potentially through class weighting or cost-sensitive learning approaches.