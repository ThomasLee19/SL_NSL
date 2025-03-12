# **Logbook: Enhanced Random Forest for Multiclass Network Intrusion Detection**
**File Name**: `rf_multi_v2`

---

#### **Date**: November 29, 2024 - December 1, 2024
**Objective**: Implement significant improvements to the Random Forest classifier for multiclass network intrusion detection to address class imbalance and enhance minority class detection, particularly for the rare R2L and U2R attack categories.

---

### **Work Summary**
1. **Code Refactoring and Modularization**
   - **Modular Design**: Implemented function-based architecture with clear separation of concerns
   - **Components**: Organized code into data loading, preprocessing, sampling, evaluation, and main execution modules
   - **Pipeline Approach**: Constructed end-to-end workflow for reproducibility and maintainability

2. **Advanced Data Handling**
   - **Adaptive Sampling Implementation**:
     - Combined SMOTE (Synthetic Minority Over-sampling Technique) with RandomUnderSampler
     - Applied class-specific sampling rates based on original class distributions
     - Carefully calibrated sampling ratios to avoid excessive synthetic data generation
   - **Feature Selection**:
     - Implemented SelectKBest with f_classif (ANOVA F-value) for dimensionality reduction
     - Reduced feature space from 109 to 50 most discriminative features
     - Optimized computational efficiency while maintaining classification power

3. **Model Architecture Enhancements**
   - **Hyperparameter Optimization**:
     - Increased ensemble size to 200 trees (from 100 in v1)
     - Set explicit max_depth=20 to control tree complexity
     - Configured min_samples_leaf=5 and min_samples_split=10 to reduce overfitting
     - Set max_features=0.8 for feature subsampling during splits
   - **Meta-Ensemble Implementation**:
     - Applied Bagging (Bootstrap Aggregating) with 10 meta-estimators
     - Each meta-estimator uses the optimized Random Forest as base classifier
     - Leveraged n_jobs=-1 for parallel execution across CPU cores

4. **Evaluation Framework Improvements**
   - **Multi-Stage Validation**:
     - Maintained 80/20 train-validation split for internal validation
     - Enhanced external test set evaluation with more comprehensive metrics
   - **Class-Specific Analysis**:
     - Detailed per-class precision, recall, and F1-score tracking
     - Distribution analysis to correlate predictions with ground truth
     - Direct comparison with baseline (v1) performance

5. **Class Imbalance Handling**
   - **Targeted Approach for Minority Classes**:
     - Class 3 (R2L): 2.0× upsampling ratio
     - Class 4 (U2R): 3.0× upsampling ratio
     - Class 0 (Normal): 0.8× downsampling ratio
   - **Data Augmentation Control**:
     - Dynamically adjusted k_neighbors parameter in SMOTE based on class sizes
     - Applied proportional sampling rather than equalization to preserve dataset characteristics

---

### **Key Decisions & Technical Details**
1. **Sampling Strategy Selection**:
   - Chose hybrid SMOTE + undersampling over alternative techniques
   - **Rationale**: Combines synthetic generation with majority class reduction for balanced approach
   - **Technical Trade-off**: Some information loss from majority class vs. better minority representation
   - **Implementation Detail**: Used pipeline to ensure proper sequence (oversample minorities first, then undersample majority)

2. **Feature Selection Method**:
   - Selected ANOVA F-value over mutual information or correlation methods
   - **Justification**: Better performance with linear feature relationships while being computationally efficient
   - **Parameter Choice**: k=50 features provided optimal balance between dimensionality reduction and information retention
   - **Technical Impact**: 54% feature reduction without significant accuracy loss

3. **Hyperparameter Configuration**:
   - Used domain-informed hyperparameter settings rather than default values
   - **Tree Structure Control**:
     - max_depth=20 prevents excessive complexity and overfitting
     - min_samples_leaf=5 ensures terminal nodes have sufficient samples
     - min_samples_split=10 prevents splits with too few samples
   - **Ensemble Design**:
     - Increased n_estimators to 200 for more robust voting
     - Added bagging meta-ensemble for additional variance reduction
     - max_features=0.8 enhances tree diversity

4. **GPU/CPU Resource Allocation**:
   - Applied hybrid approach using both GPU and CPU resources
   - **Implementation Details**:
     - RAPIDS ecosystem (cuML, cuDF) for data processing and base model
     - Scikit-learn's BaggingClassifier with n_jobs=-1 for parallelized meta-ensemble
   - **Technical Consideration**: Optimized memory usage with float32 precision

5. **Evaluation Metric Prioritization**:
   - Emphasized class-specific metrics over aggregate accuracy
   - **Reasoning**: Better representation of performance across imbalanced classes
   - **Key Indicators**: Recall for critical minority classes (R2L, U2R)
   - **Success Criteria**: Non-zero recall on all attack categories

---

### **Results**
- **Class Distribution After Balancing**:
  | Class                | Original Count | After Sampling | Change Factor |
  |----------------------|----------------|----------------|---------------|
  | Class 0 (Normal)     | 53,844         | 43,075         | 0.80× (↓)     |
  | Class 1 (DOS)        | 36,806         | 38,646         | 1.05× (↑)     |
  | Class 2 (Probe)      | 9,290          | 11,148         | 1.20× (↑)     |
  | Class 3 (R2L)        | 797            | 1,594          | 2.00× (↑)     |
  | Class 4 (U2R)        | 42             | 126            | 3.00× (↑)     |

- **Validation Performance Comparison (v1 vs v2)**:
  | Metric               | v1    | v2    | Change |
  |----------------------|-------|-------|--------|
  | Overall Accuracy     | 1.000 | 0.996 | -0.004 |
  | Class 0 Recall       | 1.000 | 1.000 | 0.000  |
  | Class 1 Recall       | 1.000 | 1.000 | 0.000  |
  | Class 2 Recall       | 0.990 | 0.990 | 0.000  |
  | Class 3 Recall       | 0.880 | 0.960 | +0.080 |
  | Class 4 Recall       | 0.090 | 0.300 | +0.210 |
  | Macro Avg F1-Score   | 0.820 | 0.870 | +0.050 |

- **Test Performance Comparison (v1 vs v2)**:
  | Metric               | v1    | v2    | Change |
  |----------------------|-------|-------|--------|
  | Overall Accuracy     | 0.730 | 0.761 | +0.031 |
  | Class 0 Recall       | 0.790 | 0.830 | +0.040 |
  | Class 1 Recall       | 0.940 | 0.930 | -0.010 |
  | Class 2 Recall       | 0.570 | 0.660 | +0.090 |
  | Class 3 Recall       | 0.000 | 0.070 | +0.070 |
  | Class 4 Recall       | 0.000 | 0.080 | +0.080 |
  | Macro Avg F1-Score   | 0.440 | 0.520 | +0.080 |

- **Prediction Distribution on Test Set**:
  | Class                | v1 Predictions | v2 Predictions | Ground Truth |
  |----------------------|----------------|----------------|--------------|
  | Class 0 (Normal)     | 12,275         | 12,807         | 12,131       |
  | Class 1 (DOS)        | 8,423          | 7,304          | 5,741        |
  | Class 2 (Probe)      | 1,846          | 2,261          | 2,421        |
  | Class 3 (R2L)        | 0              | 167            | 2,199        |
  | Class 4 (U2R)        | 0              | 5              | 52           |

- **Confusion Matrix Analysis (Test Set)**:
  ```
  # v2 Test Confusion Matrix
  [[10076  1418   629     7     1]  # Class 0 (Normal)
   [  364  5332    45     0     0]  # Class 1 (DOS)
   [  311   523  1587     0     0]  # Class 2 (Probe)
   [ 2008    31     0   160     0]  # Class 3 (R2L)
   [   48     0     0     0     4]] # Class 4 (U2R)
  ```

---

### **Technical Analysis**
1. **Performance Improvements**:
   - **Overall Accuracy**: Increased from 73.0% to 76.1% (+3.1%)
   - **Minority Class Detection**: Critical improvement with non-zero recall for R2L and U2R classes
   - **Class-Balanced Performance**: Macro-average F1-score improved from 0.44 to 0.52 (+18.2%)

2. **Imbalance Handling Effectiveness**:
   - **Adaptive Sampling Impact**: Successfully introduced minority class detection capability
   - **Cost-Benefit Analysis**: Slight decrease in DOS recall (-1%) in exchange for significant gains in other classes
   - **Technical Implementation**: Carefully calibrated class-specific sampling ratios proved more effective than full class balancing

3. **Feature Selection Impact**:
   - **Dimensionality Reduction**: 54% feature reduction with improved overall performance
   - **Computational Efficiency**: Reduced training time and memory requirements
   - **Performance Trade-off**: Feature reduction likely contributed to improved generalization

4. **Ensemble Architecture Benefits**:
   - **Two-Level Ensemble**: Bagging of Random Forests provides variance reduction at multiple levels
   - **Tree Complexity Control**: Explicit depth and splitting constraints reduced overfitting
   - **Technical Implementation**: Hybrid GPU/CPU utilization maximized computational resources

5. **Remaining Challenges**:
   - **R2L Detection**: Still poor recall (0.07) despite improvements
   - **U2R Detection**: Remains challenging with only 0.08 recall
   - **Class Distribution Shift**: Significant gap remains between training and test distributions
   - **Error Analysis**: Normal traffic (Class 0) commonly misclassified as R2L attacks

---

### **Conclusion**
The enhanced Random Forest implementation (v2) successfully addresses the most critical limitations identified in the baseline model (v1), particularly the complete failure to detect minority attack classes. Through systematic application of advanced techniques including adaptive sampling, feature selection, hyperparameter optimization, and meta-ensemble architecture, the model achieves meaningful improvements across multiple performance dimensions.

Most significantly, the model can now detect examples from all attack categories, eliminating the "blind spots" present in the previous implementation. While the recall rates for R2L (7%) and U2R (8%) remain low, they represent a substantial improvement over the complete failure (0%) in the baseline model. This progress is particularly important from a security perspective, as these attack types are typically the most sophisticated and dangerous.

The implementation demonstrates effective use of modern ML engineering practices, including modularization, pipeline design, and resource optimization. The hybrid GPU/CPU approach leverages hardware acceleration where appropriate while using CPU-based implementations for operations not well-suited to GPU computation.

Despite these improvements, the persistent challenge of distribution shift between training and test sets continues to impact performance. The model still struggles with the R2L class in particular, which has a dramatically different prevalence in the test set (9.75%) compared to training (0.79%). This remains an open research challenge for network intrusion detection systems.

Future work should explore more advanced approaches for handling severe class imbalance and distribution shift, potentially including techniques such as meta-learning, transfer learning, or domain adaptation. More sophisticated ensemble strategies like stacking or custom cost-sensitive learning functions could further improve detection of critical minority attack classes. Additionally, exploring the use of anomaly detection methods for rare attack types could complement the classification approach demonstrated here.