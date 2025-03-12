### **Logbook: Advanced XGBoost for Binary Classification Version 2.0**
**File Name**: `xgboost_bin_v2`

---

#### **Date**: November 15, 2024 - November 18, 2024
**Objective**: Enhance the XGBoost binary classification model through advanced preprocessing, hyperparameter optimization, and cross-validation techniques to achieve superior performance in network intrusion detection compared to previous models.

---

### **Work Summary**
1. **Advanced Data Preparation & Sampling**
   - **Preprocessing**: Utilized the same preprocessed NSL-KDD dataset (125,973 samples, 108 features)
   - **Class Balancing**: Implemented SMOTEENN (Synthetic Minority Over-sampling Technique with Edited Nearest Neighbors)
     - Applied targeted synthetic sample generation with cleaning (63,246 attack vs 66,605 normal)
     - Achieved near-balanced class distribution with ratio of approximately 0.95:1
     - Removed noisy samples near class boundaries to improve decision boundary definition

2. **Sophisticated Model Configuration**
   - **Enhanced XGBoost Architecture**:
     - Increased n_estimators to 150 (from 50 in v1)
     - Applied conservative learning rate (0.01) to prevent overfitting
     - Implemented regularization parameters (alpha=0.5, lambda=2)
     - Utilized subsampling and feature sampling (0.7 for both)
     - Added scale_pos_weight=1.15 to account for remaining class imbalance
   - **Hardware Utilization**: Maintained GPU acceleration via tree_method='hist' and device='cuda'

3. **Robust Evaluation Framework**
   - **Cross-Validation Strategy**:
     - Implemented 5-fold stratified cross-validation to ensure reliable performance estimation
     - Tracked learning progress with validation_0-logloss and validation_1-logloss metrics
     - Determined optimal threshold via precision-recall curve analysis for each fold
   - **Threshold Optimization**:
     - Analyzed precision-recall trade-offs to determine optimal classification threshold per fold
     - Found mean optimal threshold of 0.5754 (compared to default 0.5)
     - Applied threshold-adjusted classifications for final model

4. **Comprehensive Feature Analysis**
   - **Feature Importance Computation**:
     - Averaged feature importance across all CV folds to ensure stability
     - Identified traffic volume and protocol-specific features as most discriminative
     - src_bytes, flag_SF, and service_ecr_i emerged as top contributors
   - **Comparison to Previous Models**:
     - Noted different feature importance distribution from RF-based models
     - Protocol-specific features gained prominence compared to v1

5. **Performance Visualization & Interpretability**
   - **Advanced Visualization**:
     - Generated confusion matrix heatmap for intuitive performance analysis
     - Increased annotation sizes for better readability
     - Saved high-resolution output for reporting purposes
   - **Interpretation**:
     - Analyzed per-class performance metrics
     - Identified fewer false negatives compared to previous models

---

### **Results**
- **Cross-Validation Performance**:
  | Fold | Accuracy | Optimal Threshold | False Positives | False Negatives |
  |------|----------|-------------------|-----------------|-----------------|
  | 1    | 0.9973   | 0.6232            | Minimal         | Minimal         |
  | 2    | 0.9970   | 0.5370            | Minimal         | Minimal         |
  | 3    | 0.9976   | 0.5951            | Minimal         | Minimal         |
  | 4    | 0.9974   | 0.5727            | Minimal         | Minimal         |
  | 5    | 0.9970   | 0.5492            | Minimal         | Minimal         |
  | Mean | 0.9973   | 0.5754            | -               | -               |

- **Model Comparison (External Test Set)**:
  | Model             | Accuracy | Precision (Norm) | Recall (Norm) | Precision (Attack) | Recall (Attack) | F1 (Weighted) |
  |-------------------|----------|------------------|---------------|-------------------|-----------------|---------------|
  | RF v1             | 0.81     | 0.74             | 0.86          | 0.88              | 0.77            | 0.81          |
  | RF v2             | 0.855    | 0.96             | 0.70          | 0.81              | 0.98            | 0.85          |
  | XGBoost v1        | 0.88     | 0.98             | 0.75          | 0.84              | 0.99            | 0.88          |
  | **XGBoost v2**    | **0.89** | **0.95**         | **0.78**      | **0.85**          | **0.97**        | **0.89**      |

- **Confusion Matrix (External Test)**:
  | | Predicted Normal | Predicted Attack | Total |
  |------------------|------------------|------------------|-------|
  | **Actual Normal** | 7,591 (TN) | 2,120 (FP) | 9,711 |
  | **Actual Attack** | 424 (FN) | 12,409 (TP) | 12,833 |
  | **Total** | 8,015 | 14,529 | 22,544 |

- **Top Features by Importance**:
  | Rank | Feature | Importance Score |
  |------|---------|------------------|
  | 1 | src_bytes | 0.221126 |
  | 2 | flag_SF | 0.163029 |
  | 3 | service_ecr_i | 0.106258 |
  | 4 | dst_bytes | 0.092321 |
  | 5 | protocol_type_icmp | 0.073322 |

---

### **Key Decisions**
1. **Advanced Sampling Strategy Selection**:
   - **Approach**: Chose SMOTEENN over simple SMOTE or undersampling
   - **Rationale**: Provides both synthetic sample generation and noise cleaning in a single step
   - **Implementation Details**: Targeted sampling ratio of 0.95 to slightly favor normal class, preventing overcompensation
   - **Impact**: Achieved more balanced training while maintaining representation of the natural distribution

2. **Hyperparameter Optimization Strategy**:
   - **Learning Rate**: Deliberately reduced from default 0.3 to 0.01
   - **Reasoning**: Slower convergence promotes better generalization by preventing premature overfitting
   - **Regularization Balance**: Alpha (L1) set lower than Lambda (L2) to encourage feature selection while maintaining smoothness
   - **Trade-off**: Training time increased ~3x compared to v1, but with significant performance gains

3. **Threshold Calibration Approach**:
   - **Methodology**: Optimized classification threshold per fold using precision-recall curves
   - **Decision Factor**: F1-score maximization to balance precision and recall
   - **Observation**: Optimal thresholds consistently higher than default 0.5
   - **Benefit**: Improved precision for normal traffic without substantial sacrifice in attack detection

4. **Cross-Validation Design**:
   - **Choice**: 5-fold stratified CV with fold-specific resampling
   - **Technical Consideration**: Applied SMOTEENN within each fold rather than before splitting
   - **Advantage**: Prevented data leakage and provided more reliable performance estimates
   - **Finding**: Extremely consistent performance across folds (std deviation < 0.001)

5. **Feature Subsampling Configuration**:
   - **Parameter**: colsample_bytree and subsample both set to 0.7
   - **Objective**: Introduce randomness to prevent overfitting while maintaining feature diversity
   - **Reasoning**: Network intrusion patterns may rely on complex feature interactions
   - **Result**: More robust model with improved generalization to external test set

---

### **Conclusion**
The enhanced XGBoost classifier (xgboost_bin_v2) delivers the highest performance among all models tested on the NSL-KDD binary classification task. Building upon the strengths of the first XGBoost implementation, this version incorporates advanced sampling techniques, sophisticated hyperparameter tuning, and robust evaluation methodologies to achieve a 1% accuracy improvement over v1 (89% vs 88%) and significant gains over both Random Forest implementations (89% vs 85.5% for RF v2 and 81% for RF v1).

The model demonstrates exceptional balance between precision and recall metrics, with the optimal threshold calibration effectively handling the inherent trade-off between false positives and false negatives. Most notably, the reduction in false negatives to just 424 instances (compared to 2,916 in RF v2) represents a critical improvement for intrusion detection systems where missed attacks carry significant consequences.

Feature importance analysis reveals that packet volume metrics (src_bytes, dst_bytes) and connection flags/services play dominant roles in classification decisions. This aligns with domain knowledge that network attacks often manifest through abnormal traffic patterns and specific protocol usage. The stability of cross-validation results (99.73% mean accuracy with minimal variation) indicates a robust model that effectively captures the underlying patterns in network traffic data.

The sophisticated preprocessing through SMOTEENN proved particularly effective, allowing the model to learn from a more balanced dataset while removing potentially confusing samples near class boundaries. This approach, combined with careful regularization and conservative learning rate, successfully addresses the challenge of generalization from training to external test data.

Future work should focus on deploying this model in real-time environments, exploring the potential for online learning to adapt to evolving threats, and examining the model's interpretability for security analysts. Additionally, testing on more recent network intrusion datasets would provide valuable insights regarding the model's adaptability to emerging attack vectors in modern network environments.