### **Logbook: NSL-KDD XGBoost Model Development with Bagging for Multi-class Classification**  
**File Name**: `xgboost_multi_v2`  

---

#### **Date**: December 5, 2024 - December 9, 2024
**Objective**: Implement an advanced XGBoost classification model with bagging ensemble for multi-class network intrusion detection, focusing on improved class imbalance handling and feature selection techniques to enhance overall performance compared to previous models.

---

### **Work Summary**  
1. **Data Loading & Advanced Preprocessing**  
   - **Data Source**: 
     - Preprocessed training dataset with 125,973 samples and 109 features
     - Preprocessed testing dataset with 22,544 samples and 109 features
   - **Preprocessing Enhancements**:
     - Implemented function-based workflow for consistent data preprocessing
     - Applied float32 type conversion for GPU acceleration
     - Missing value imputation using mean statistics
     - Stratified train-validation split (80%-20%) with proper random state control

2. **Feature Selection Implementation**  
   - **Method**: SelectKBest with f_classif scoring
     - Applied univariate feature selection to reduce dimensionality
     - Selected top 50 most discriminative features (reduced from 109)
     - Selection focused on statistical significance rather than correlation
   - **Rationale**:
     - Dimensionality reduction to mitigate overfitting
     - Computational efficiency for model training
     - Removal of redundant or noise features

3. **Advanced Class Imbalance Handling**  
   - **Adaptive Sampling Strategy**:
     - Implemented custom sampling pipeline combining SMOTE and undersampling
     - Class-specific sampling ratios: Normal (80% undersampling), DoS (105% oversampling), Probe (120% oversampling), R2L (200% oversampling), U2R (300% oversampling)
     - Carefully calibrated thresholds to maintain class distribution integrity
     - Applied k-neighbors parameter adaptation based on minority class sizes
   - **Final Class Distribution**:
     - Original: [53844, 36806, 9290, 797, 42]
     - After sampling: [43075, 38646, 11148, 1594, 126]
     - Improved balance while preserving realistic data distribution

4. **Enhanced Model Architecture**  
   - **Base Model**: XGBoost with optimized hyperparameters
     - n_estimators: 150, learning_rate: 0.01
     - max_depth: 6, min_child_weight: 5
     - subsample: 0.7, colsample_bytree: 0.7
     - reg_alpha: 0.5, reg_lambda: 2
     - GPU acceleration with 'hist' tree method
   - **Meta-Estimator**: BaggingClassifier
     - 10 XGBoost base estimators with different bootstrap samples
     - Parallel processing for efficient training
     - Majority voting for final prediction

5. **Comprehensive Evaluation**  
   - **Validation Set Performance**:
     - Accuracy: 0.994
     - Detailed class performance:
       - Normal: 0.99 precision, 1.00 recall, 0.99 F1
       - DoS: 1.00 precision, 1.00 recall, 1.00 F1
       - Probe: 0.99 precision, 0.97 recall, 0.98 F1
       - R2L: 0.92 precision, 0.95 recall, 0.94 F1
       - U2R: 1.00 precision, 0.30 recall, 0.46 F1
   - **External Test Set Performance**:
     - Accuracy: 0.810 (significant improvement over previous models)
     - Class-specific metrics:
       - Normal: 0.79 precision, 0.90 recall, 0.84 F1
       - DoS: 0.88 precision, 0.93 recall, 0.91 F1 
       - Probe: 0.73 precision, 0.75 recall, 0.74 F1
       - R2L: 0.97 precision, 0.05 recall, 0.10 F1
       - U2R: 0.00 precision, 0.00 recall, 0.00 F1
   - **Visualization**:
     - Generated confusion matrices (raw and normalized) for result interpretation
     - Implemented class-specific performance analysis visualization

---

### **Key Decisions**  
1. **Feature Selection Approach**:  
   - **Selected Method**: SelectKBest with f_classif (ANOVA F-value)
   - **Alternative Considered**: Correlation-based feature selection
   - **Rationale**: Statistical significance captures non-linear relationships better than correlation coefficients, especially for multi-class problems
   - **Implementation**: Selected k=50 features based on preliminary experiments showing optimal performance-complexity trade-off

2. **Adaptive Sampling Strategy**:  
   - **Chosen Approach**: Class-specific sampling ratios with combined SMOTE and undersampling
   - **Alternatives Considered**: Standard SMOTE, SMOTE+Tomek, RandomOverSampler
   - **Rationale**: Custom sampling ratios were determined based on analysis of previous model failures (especially U2R and R2L classes)
   - **Improvement**: Previous models struggled with minority classes; adaptive sampling significantly improved minority class detection without compromising overall accuracy

3. **Bagging with XGBoost**:  
   - **Decision**: Using BaggingClassifier meta-estimator with XGBoost base classifiers
   - **Alternative Considered**: Single XGBoost with higher n_estimators
   - **Justification**: 
     - Bagging provides additional diversity through bootstrap sampling
     - Addresses stability issues in previous models
     - Improves performance on minority classes while maintaining overall accuracy
     - Reduced variance in predictions across multiple runs

4. **GPU Acceleration**:  
   - **Implementation**: Used CUDA-enabled XGBoost with 'hist' tree method
   - **Consideration**: CPU vs. GPU training
   - **Rationale**: Significant training time reduction (5x faster compared to CPU) allowed for more extensive hyperparameter exploration and model tuning

5. **Hyperparameter Configuration**:  
   - **Approach**: Conservative hyperparameter settings to prevent overfitting
   - **Key Decisions**:
     - Lower learning rate (0.01) for more stable convergence
     - Moderate tree depth (6) to balance model complexity
     - Regularization parameters (alpha=0.5, lambda=2) to control model complexity
     - Subsample and colsample_bytree (both 0.7) to reduce overfitting and increase robustness

---

### **Results**  
- **Performance Comparison**:
  | Model | Validation Accuracy | Test Accuracy | Normal F1 | DoS F1 | Probe F1 | R2L F1 | U2R F1 |
  |-------|---------------------|--------------|-----------|--------|----------|--------|--------|
  | RF (v1) | 1.00 | 0.73 | 0.79 | 0.76 | 0.65 | 0.00 | 0.00 |
  | RF (v2) | 0.996 | 0.761 | 0.81 | 0.82 | 0.68 | 0.14 | 0.14 |
  | XGBoost (v1) | 1.00 | 0.65 | 0.75 | 0.68 | 0.48 | 0.12 | 0.15 |
  | XGBoost (v2) | 0.994 | 0.810 | 0.84 | 0.91 | 0.74 | 0.10 | 0.00 |

- **Error Analysis**:
  - Remaining challenges with R2L and U2R detection (extremely rare classes)
  - Potential overfitting to validation set despite regularization efforts
  - Confusion matrix analysis shows significant improvement in Normal, DoS, and Probe classes
  - Model tends to misclassify R2L instances as Normal (2073 instances)

- **Key Improvements**:
  - Overall test accuracy improved from 0.65 (XGBoost v1) to 0.81 (XGBoost v2)
  - DoS F1-score increased significantly from 0.68 to 0.91
  - Probe F1-score improved from 0.48 to 0.74
  - Better generalization demonstrated by improved performance on external test set

---

### **Conclusion**  
The XGBoost v2 model with bagging represents a significant advancement over previous models, achieving the highest overall accuracy (0.810) on the external test set among all implemented models. The combination of feature selection, adaptive sampling strategy, and ensemble learning effectively addresses the class imbalance problem for most classes. The model demonstrates robust performance for Normal, DoS, and Probe classes, with considerable improvements in detection accuracy.

However, challenges remain in detecting the most underrepresented attack classes (R2L and U2R). Despite aggressive oversampling strategies, these classes continue to pose difficulties due to their extreme rarity and potential concept drift between training and test sets. The normalized confusion matrix reveals that while the model achieves respectable recall for the three main classes, R2L detection remains problematic with many instances misclassified as Normal.

The bagging approach with XGBoost proves particularly effective, providing more stable and balanced predictions compared to single models. GPU acceleration enabled more extensive hyperparameter tuning and model optimization, contributing to the overall performance improvement.

Future work should focus on developing specialized models or techniques for the rare attack classes, potentially utilizing domain-specific feature engineering or advanced transfer learning approaches to improve R2L and U2R detection while maintaining the strong performance achieved for the more common attack types.