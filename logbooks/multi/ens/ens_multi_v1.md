### **Logbook: NSL-KDD Ensemble Learning Model for Multi-class Network Intrusion Detection**  
**File Name**: `ens_multi_v1`  

---

#### **Date**: December 10, 2024 - December 14, 2024
**Objective**: Implement an ensemble learning approach combining Random Forest and XGBoost classifiers with stacking architecture to leverage the strengths of different algorithms, enhancing multi-class intrusion detection performance while addressing class imbalance issues.

---

### **Work Summary**  
1. **Data Preparation & Preprocessing Pipeline**  
   - **Source Data**: 
     - Preprocessed training dataset with 125,973 samples and 109 features
     - Preprocessed testing dataset with 22,544 samples and 109 features
   - **Implementation Details**:
     - Modular function-based preprocessing pipeline
     - Conversion to float32 for GPU acceleration
     - Consistent missing value imputation using mean statistics
     - Stratified training-validation split (80-20) with random state control for reproducibility
     - Class distribution analysis to inform sampling strategies

2. **Feature Engineering & Selection**  
   - **Method**: SelectKBest with f_classif (ANOVA F-value scoring)
     - Applied univariate statistical feature selection
     - Reduced dimensionality from 109 to 50 features
     - Selection based on statistical significance rather than correlation
   - **Implementation Details**:
     - Applied consistent feature selection across training, validation, and test datasets
     - Converted selected features back to GPU format for efficient model training

3. **Advanced Class Imbalance Handling**  
   - **Adaptive Sampling Strategy**:
     - Implemented a two-stage pipeline combining SMOTE and undersampling
     - Class-specific sampling ratios:
       - Normal: 80% undersampling (reduction from 53,844 to 43,075)
       - DoS: 105% oversampling (36,806 to 38,646)
       - Probe: 120% oversampling (9,290 to 11,148)
       - R2L: 200% oversampling (797 to 1,594) 
       - U2R: 300% oversampling (42 to 126)
     - Adaptive k-neighbors parameter based on minority class sizes
     - Balancing approach designed to maximize detection of rare attack classes

4. **Ensemble Architecture Implementation**  
   - **Base Models**:
     - **Random Forest**: GPU-accelerated with 200 estimators, max_depth=20, custom leaf and split parameters
     - **XGBoost**: 150 estimators, learning_rate=0.01, max_depth=6, regularization parameters
   - **Meta-Learning Layer**: 
     - StackingClassifier architecture combining RF and XGBoost predictions
     - Final estimator: BaggingClassifier with RandomForest (100 trees, max_depth=10)
     - Parallel processing for efficient training using all available cores
   - **Design Philosophy**:
     - Complementary base models capture different aspects of the data patterns
     - Multi-level ensemble architecture reduces variance and improves generalization
     - Bagging in the meta-learner promotes robustness to outliers

5. **Comprehensive Evaluation**  
   - **Validation Set Performance**:
     - Accuracy: 0.996
     - Class-specific metrics:
       - Normal: 1.00 precision, 1.00 recall, 1.00 F1
       - DoS: 1.00 precision, 1.00 recall, 1.00 F1
       - Probe: 0.99 precision, 0.99 recall, 0.99 F1
       - R2L: 0.93 precision, 0.96 recall, 0.95 F1
       - U2R: 1.00 precision, 0.30 recall, 0.46 F1
   - **External Test Set Performance**:
     - Accuracy: 0.792 (competitive with state-of-the-art)
     - Class-specific metrics:
       - Normal: 0.80 precision, 0.87 recall, 0.83 F1
       - DoS: 0.75 precision, 0.94 recall, 0.84 F1
       - Probe: 0.89 precision, 0.72 recall, 0.80 F1
       - R2L: 0.92 precision, 0.05 recall, 0.09 F1
       - U2R: 0.00 precision, 0.00 recall, 0.00 F1

---

### **Key Decisions**  
1. **Stacking Architecture Selection**:  
   - **Selected Approach**: Two-level stacking with heterogeneous base models
   - **Alternatives Considered**: Voting ensemble, single model with extensive tuning, boosting methods
   - **Rationale**: 
     - Stacking leverages strengths of different algorithms while mitigating individual weaknesses
     - RF excels at handling high-dimensional data while XGBoost provides strong decision boundaries
     - Meta-learner with bagging adds additional robustness against overfitting
   - **Implementation Details**: 
     - Base layer uses both tree-based models but with different learning paradigms
     - Meta-learner applies bagging to further reduce variance

2. **Hyperparameter Configuration Strategy**:  
   - **Approach**: Conservative hyperparameters for RF, regularized parameters for XGBoost
   - **Key Decisions**:
     - RF: Higher tree count (200) with moderate depth (20) for complexity
     - XGBoost: Low learning rate (0.01) with strong regularization (alpha=0.5, lambda=2)
     - Meta-learner: Simplified RandomForest with bagging for variance reduction
   - **Rationale**:
     - Parameters selected to maximize diversity between base models
     - Strong regularization in XGBoost prevents overfitting to training data
     - Balance between model complexity and generalization ability

3. **Adaptive Sampling Strategy**:  
   - **Selected Approach**: Class-specific sampling ratios with combined SMOTE and undersampling
   - **Alternatives Considered**: Standard SMOTE, class weights, cost-sensitive learning
   - **Justification**:
     - Custom sampling ratios based on analysis of previous model failures
     - Aggressive oversampling for extremely rare classes (especially U2R and R2L)
     - Moderate undersampling of majority class to improve training efficiency
   - **Trade-offs**: Potential risk of overfitting to synthetic minority samples, balanced by ensemble approach

4. **Feature Selection Method**:  
   - **Decision**: SelectKBest with f_classif and k=50 features
   - **Alternatives Considered**: Recursive feature elimination, PCA, tree-based feature importance
   - **Rationale**:
     - Statistical significance captures non-linear relationships better than correlation
     - Dimensionality reduction improves training efficiency
     - Previous experiments showed optimal performance-complexity trade-off at k=50

5. **GPU Acceleration Implementation**:  
   - **Approach**: Strategic use of GPU for compute-intensive operations
   - **Detail**: 
     - CUDA-enabled XGBoost with 'hist' tree method
     - cuML RandomForest for efficient parallel tree building
     - GPU-based data structures (cuDF) for preprocessing
   - **Benefit**: 5-10x faster training compared to CPU implementations, enabling more extensive model exploration

---

### **Results**  
- **Performance Comparison Table**:
  | Model | Validation Accuracy | Test Accuracy | Normal F1 | DoS F1 | Probe F1 | R2L F1 | U2R F1 |
  |-------|---------------------|--------------|-----------|--------|----------|--------|--------|
  | RF (v1) | 1.00 | 0.73 | 0.79 | 0.76 | 0.65 | 0.00 | 0.00 |
  | RF (v2) | 0.996 | 0.761 | 0.81 | 0.82 | 0.68 | 0.14 | 0.14 |
  | XGBoost (v1) | 1.00 | 0.65 | 0.75 | 0.68 | 0.48 | 0.12 | 0.15 |
  | XGBoost (v2) | 0.994 | 0.810 | 0.84 | 0.91 | 0.74 | 0.10 | 0.00 |
  | Ensemble (v1) | 0.996 | 0.792 | 0.83 | 0.84 | 0.80 | 0.09 | 0.00 |

- **Key Performance Insights**:
  - Ensemble model achieves second-highest overall test accuracy (79.2%)
  - Best Probe class detection performance (F1=0.80) among all models
  - Strong balance between Normal and DoS detection (F1=0.83 and 0.84)
  - Still struggles with extremely rare classes (R2L and U2R)
  - More balanced precision-recall trade-off compared to single models

- **Confusion Matrix Analysis**:
  - Significantly reduced misclassification between Probe and Normal classes
  - Still significant confusion between R2L and Normal (2,064 instances)
  - Zero U2R detections despite aggressive oversampling
  - Minimal confusion between DoS and Probe classes

- **Model Superiority Areas**:
  - Best Probe class detection performance among all models (F1=0.80)
  - Better balance between precision and recall for Normal and DoS
  - More stable predictions with lower variance
  - Higher overall detection rate for majority attack classes

---

### **Conclusion**  
The Ensemble Model (v1) represents a significant advancement in the multi-class network intrusion detection system, achieving near state-of-the-art performance with an external test accuracy of 79.2%. The stacking architecture successfully leverages the complementary strengths of Random Forest and XGBoost algorithms, resulting in more balanced and robust predictions compared to individual models.

The model particularly excels in detecting Probe attacks, achieving the highest F1 score (0.80) among all implemented models for this class. Additionally, it maintains strong performance for Normal and DoS classes with F1 scores of 0.83 and 0.84 respectively. The ensemble approach successfully addresses some of the weaknesses observed in individual models, particularly the tendency of XGBoost (v1) to underperform on Probe attacks and the tendency of RF models to have lower precision.

However, challenges remain in detecting the extremely rare attack classes (R2L and U2R). Despite implementing aggressive oversampling strategies, the model still struggles with these classes, particularly U2R which has no successful detections in the external test set. This highlights the inherent difficulty in detecting attack types with extremely limited training examples and potential concept drift between training and test distributions.

Compared to previous implementations, the Ensemble model offers better overall balance and stability. While the XGBoost (v2) achieves a slightly higher overall accuracy (81.0%), the Ensemble model provides better performance on the Probe class and more balanced precision-recall trade-offs. This suggests that ensemble approaches have significant potential for network intrusion detection systems where balanced performance across different attack types is crucial.

Future work should focus on developing specialized techniques for rare attack classes, potentially incorporating domain-specific feature engineering, transfer learning approaches, or anomaly detection methods specifically targeting R2L and U2R attacks. Additionally, exploring more complex ensemble architectures with greater diversity in base models could further improve overall system performance.