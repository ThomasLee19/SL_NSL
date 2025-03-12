### **Logbook: Enhanced Supervised Binary Classification with Random Forest Version 2.0**
**File Name**: `rf_bin_v2`

---

#### **Date**: November 5, 2024 - November 10, 2024
**Objective**: Enhance the existing Random Forest classifier for binary network intrusion detection by implementing advanced sampling techniques, hyperparameter optimization, and cross-validation to improve model generalization and performance on external test data.

---

### **Work Summary**
1. **Model Configuration Enhancement**
   - **Base Architecture**: GPU-accelerated Random Forest using RAPIDS cuML
   - **Initial Hyperparameters**:
     - n_estimators: 100 (increased from 50 in v1)
     - max_depth: 10 (explicit control vs. unlimited in v1)
     - min_samples_leaf: 10 (new parameter)
     - min_samples_split: 20 (new parameter)
     - max_features: 'sqrt' (new parameter)
     - n_bins: 256 (GPU-specific optimization parameter)

2. **Advanced Training Strategy Implementation**
   - **Class Imbalance Handling**:
     - Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance class distribution
     - Improved representation of minority attack patterns in training data
   - **Hyperparameter Optimization**:
     - Implemented comprehensive grid search across 480 parameter combinations
     - Used weighted F1-score as optimization metric
     - Managed GPU memory constraints with manual search implementation
   - **Cross-Validation**:
     - Utilized 10-fold Stratified Cross-Validation to ensure stable performance assessment
     - Maintained class distribution consistency across folds

3. **Grid Search Parameter Space**
   - **Tree Structure Parameters**:
     - n_estimators: [50, 100, 150, 200]
     - max_depth: [8, 10, 12, 15, 20]
   - **Node Split Parameters**:
     - min_samples_split: [15, 20, 25, 30]
     - min_samples_leaf: [5, 10, 15]
   - **Feature Selection Parameters**:
     - max_features: ['sqrt', 0.3, 0.5, 0.7]
   - **GPU Optimization Parameters**:
     - n_bins: [128, 256]

4. **Evaluation Protocol Enhancement**
   - **Internal Validation**:
     - 20% stratified hold-out from training data
     - Confusion matrix and classification report analysis
   - **Cross-Validation Stability Assessment**:
     - 10-fold CV with SMOTE applied individually to each fold
     - Mean and standard deviation of F1-scores
   - **External Test Evaluation**:
     - Comprehensive metrics on separate test dataset
     - Analysis of precision-recall tradeoff

---

### **Results**
- **Optimal Hyperparameters**:
  | Parameter | rf_bin_v1 Value | rf_bin_v2 Optimal Value |
  |-----------|-----------------|-------------------------|
  | n_estimators | 50 | 100 |
  | max_depth | Unlimited | 20 |
  | min_samples_split | Default (2) | 15 |
  | max_features | Default (All) | 0.5 |
  | min_samples_leaf | Default (1) | 10 |

- **Model Performance**:
  | Metric | Internal Validation | Cross-Validation | External Test | v1 External Test |
  |--------|---------------------|------------------|---------------|------------------|
  | Accuracy | 0.998 | 0.998 (±0.001) | 0.855 | 0.81 |
  | F1-score (Class 0) | 1.00 | - | 0.81 | 0.80 |
  | F1-score (Class 1) | 1.00 | - | 0.88 | 0.82 |
  | Precision (Class 0) | 1.00 | - | 0.96 | 0.74 |
  | Recall (Class 0) | 1.00 | - | 0.70 | 0.86 |
  | Precision (Class 1) | 1.00 | - | 0.81 | 0.88 |
  | Recall (Class 1) | 1.00 | - | 0.98 | 0.77 |

- **Confusion Matrix (External Test)**:
  | | Predicted Normal | Predicted Attack |
  |------------------|------------------|------------------|
  | **Actual Normal** | 6,759 (TN) | 2,952 (FP) |
  | **Actual Attack** | 308 (FN) | 12,525 (TP) |

- **Cross-Validation Stability**:
  - Mean weighted F1-score: 0.998
  - Standard deviation: ±0.0005
  - Demonstrates consistent performance across different data subsets

---

### **Key Decisions**
1. **Class Balancing with SMOTE**:
   - **Problem Addressed**: Class imbalance in training data affecting model bias
   - **Approach**: Generated synthetic samples for minority class using nearest-neighbor interpolation
   - **Impact**: Improved recall for attack class (0.98 vs 0.77 in v1), boosting overall detection capability

2. **Hyperparameter Optimization Strategy**:
   - **Approach**: Comprehensive grid search with F1-score optimization
   - **Tradeoff**: Training time (~5x longer) vs. performance gains (+4.5% accuracy on external test)
   - **Key Finding**: Shallower trees (depth 8-20) with more restrictive split criteria outperformed default configuration

3. **Feature Selection Method**:
   - **Decision**: Used max_features=0.5 instead of all features
   - **Reasoning**: Reduced overfitting by introducing randomness in feature selection process
   - **Benefit**: Improved generalization to external test set while maintaining near-perfect validation performance

4. **Evaluation Metrics Focus**:
   - **Primary Metric**: Weighted F1-score instead of accuracy
   - **Justification**: Better representation of model performance in presence of class imbalance
   - **Result**: More balanced precision-recall tradeoff on external test set

5. **GPU Resource Management**:
   - **Challenge**: Memory constraints during grid search
   - **Solution**: Manual implementation of grid search with optimized parameter combinations
   - **Outcome**: Successfully leveraged GPU acceleration while enabling extensive hyperparameter exploration

---

### **Conclusion**
The enhanced Random Forest model (rf_bin_v2) demonstrates significant improvements over the baseline version, achieving higher accuracy (85.5% vs. 81%) and more balanced performance metrics on the external test set. The integration of SMOTE for class balancing was particularly effective, addressing the class imbalance issue and resulting in substantially improved attack detection rates (recall increased from 0.77 to 0.98 for attack class).

The extensive hyperparameter optimization revealed that a combination of moderate tree depth (20), more estimators (100), and partial feature selection (50%) yields optimal performance. This configuration balances model complexity against generalization capability, reducing overfitting observed in the baseline model.

Cross-validation results confirm the stability of the model, with consistently high performance across different data subsets (F1 = 0.998 ±0.001). However, despite these improvements, there remains a noticeable gap between validation and external test performance, suggesting some degree of dataset shift that warrants further investigation.

The precision-recall trade-off has shifted compared to v1; while v2 shows higher overall accuracy and F1-scores, it demonstrates lower recall for normal traffic (0.70 vs 0.86) but significantly higher recall for attacks (0.98 vs 0.77). This shift aligns with security priorities, as false negatives (missed attacks) are generally more costly than false positives.

Future work should explore more sophisticated feature engineering techniques, ensemble methods combining multiple classifiers, and deeper analysis of misclassified instances to further improve generalization performance on external data.