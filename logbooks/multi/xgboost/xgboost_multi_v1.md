# **Logbook: XGBoost Implementation for Multiclass Network Intrusion Detection**
**File Name**: `xgboost_multi_v1`

---

#### **Date**: December 1, 2024 - December 4, 2024
**Objective**: Implement and evaluate a GPU-accelerated XGBoost classifier for multiclass network intrusion detection, comparing its performance against previous Random Forest models while exploring gradient boosting's capabilities for handling class imbalance in the NSL-KDD dataset.

---

### **Work Summary**
1. **Implementation Setup**
   - **Libraries**: Utilized XGBoost with CUDA integration alongside RAPIDS ecosystem (cuDF)
   - **Hardware**: NVIDIA GPU for accelerated training and inference
   - **Data Source**: Reused pre-processed NSL-KDD dataset (train: 125,973 × 110, test: 22,544 × 110)

2. **Data Processing**
   - **Loading Strategy**: Direct import of preprocessed datasets using cuDF
   - **Type Conversion**: Converted features to float32 for optimized GPU computation
   - **Dataset Integrity**: Validated absence of missing values in training data
   - **Split Configuration**: Standard 80/20 train-validation split with random_state=42

3. **Model Implementation**
   - **Algorithm Selection**: XGBoost as a boosting alternative to Random Forest
   - **Core Hyperparameters**:
     - n_estimators=100 (ensemble size)
     - tree_method='hist' (histogram-based approximation)
     - device='cuda' (GPU acceleration)
     - objective='multi:softmax' (multiclass configuration)
     - num_class=5 (explicit class count specification)
   - **Gradient Boosting Approach**: Sequential tree building with gradient optimization

4. **Feature Analysis**
   - **Importance Extraction**: Leveraged XGBoost's built-in feature importance metrics
   - **Top Features**: Identified service_eco_i, service_ecr_i, and diff_srv_rate as most influential
   - **Ranking Methodology**: Used gain-based importance calculation

5. **Model Evaluation**
   - **Validation Performance**: Assessed on 20% holdout data
   - **External Testing**: Evaluated on separate test dataset
   - **Metric Analysis**: Calculated accuracy, precision, recall, F1-score by class
   - **Distribution Comparison**: Analyzed prediction distributions vs. ground truth

---

### **Key Technical Decisions**
1. **Algorithm Selection Rationale**:
   - Chose XGBoost over Random Forest for sequential boosting advantages
   - **Technical Advantage**: Gradient-based optimization potentially better for imbalanced data
   - **Implementation Detail**: No explicit class weighting needed due to boosting's adaptive nature
   - **Resource Consideration**: Utilized specialized 'hist' tree method for memory efficiency

2. **GPU Acceleration Strategy**:
   - Implemented end-to-end GPU training pipeline
   - **Configuration**: Used XGBoost's native CUDA support (device='cuda')
   - **Optimization**: Histogram-based tree construction ('hist') for reduced memory footprint
   - **Implementation Note**: XGBoost's GPU implementation handles categorical features differently than Random Forest

3. **Multiclass Configuration**:
   - Direct multiclass approach vs. binary classifiers
   - **Technical Setup**: Used softmax objective with explicit class count
   - **Alternative Considered**: One-vs-rest implementation
   - **Decision Factor**: XGBoost's native multiclass capacity with softmax loss function

4. **Default Parameter Selection**:
   - No extensive hyperparameter tuning in initial implementation
   - **Reasoning**: Establish baseline performance and compare algorithm fundamentals first
   - **Trade-off**: Potential suboptimal configuration vs. faster development cycle
   - **Future Direction**: Apply hyperparameter optimization guided by initial results

5. **Class Imbalance Handling**:
   - Relied on XGBoost's intrinsic handling of imbalanced data
   - **Technical Approach**: No explicit balancing techniques (unlike RF_v2)
   - **Alternative Considered**: SMOTE or class weighting as in RF_v2
   - **Justification**: Test gradient boosting's natural capability for minority class detection

---

### **Results & Comparative Analysis**
- **Validation Performance Across Models**:
  | Metric               | RF_v1  | RF_v2  | XGBoost | XGB vs RF_v2 |
  |----------------------|--------|--------|---------|--------------|
  | Overall Accuracy     | 1.000  | 0.996  | 1.000   | +0.004       |
  | Class 0 Recall       | 1.000  | 1.000  | 1.000   | 0.000        |
  | Class 1 Recall       | 1.000  | 1.000  | 1.000   | 0.000        |
  | Class 2 Recall       | 0.990  | 0.990  | 1.000   | +0.010       |
  | Class 3 Recall       | 0.880  | 0.960  | 0.980   | +0.020       |
  | Class 4 Recall       | 0.090  | 0.300  | 0.700   | +0.400       |
  | Macro Avg F1-Score   | 0.820  | 0.870  | 0.950   | +0.080       |

- **Test Set Performance Comparison**:
  | Metric               | RF_v1  | RF_v2  | XGBoost | XGB vs RF_v2 |
  |----------------------|--------|--------|---------|--------------|
  | Overall Accuracy     | 0.730  | 0.761  | 0.650   | -0.111       |
  | Class 0 Recall       | 0.790  | 0.830  | 0.660   | -0.170       |
  | Class 1 Recall       | 0.940  | 0.930  | 0.850   | -0.080       |
  | Class 2 Recall       | 0.570  | 0.660  | 0.680   | +0.020       |
  | Class 3 Recall       | 0.000  | 0.070  | 0.070   | 0.000        |
  | Class 4 Recall       | 0.000  | 0.080  | 0.100   | +0.020       |
  | Macro Avg F1-Score   | 0.440  | 0.520  | 0.440   | -0.080       |

- **Confusion Matrix Analysis (Test Set)**:
  ```
  # XGBoost Test Confusion Matrix
  [[8064 2351 1710    0    6]  # Class 0 (Normal)
   [ 195 4852  694    0    0]  # Class 1 (DOS)
   [ 294  487 1635    5    0]  # Class 2 (Probe)
   [ 837  886  329  145    2]  # Class 3 (R2L)
   [  19   18    7    3    5]] # Class 4 (U2R)
  ```

- **Top 5 Feature Importance**:
  | Feature                      | Importance  |
  |------------------------------|-------------|
  | service_eco_i                | 0.271069    |
  | service_ecr_i                | 0.203579    |
  | diff_srv_rate                | 0.111916    |
  | service_http                 | 0.078523    |
  | dst_host_srv_serror_rate     | 0.072739    |

- **Prediction Distribution Comparison (Test Set)**:
  | Class                | RF_v1        | RF_v2        | XGBoost      | Ground Truth |
  |----------------------|--------------|--------------|--------------|--------------|
  | Class 0 (Normal)     | 12,275       | 12,807       | 9,409        | 12,131       |
  | Class 1 (DOS)        | 8,423        | 7,304        | 8,594        | 5,741        |
  | Class 2 (Probe)      | 1,846        | 2,261        | 4,375        | 2,421        |
  | Class 3 (R2L)        | 0            | 167          | 153          | 2,199        |
  | Class 4 (U2R)        | 0            | 5            | 13           | 52           |

---

### **Technical Analysis**
1. **Validation vs. Test Performance Gap**:
   - **XGBoost**: Extreme difference between validation (1.00) and test (0.65) accuracy
   - **Comparison**: Gap larger than RF_v2 (0.996 vs 0.761)
   - **Technical Cause**: Likely stronger overfitting tendency with gradient boosting
   - **Impact**: Better minority class detection on validation but worse overall test performance

2. **Class Imbalance Handling**:
   - **Validation Set**: XGBoost significantly outperformed RF for minority classes
     - Class 4 (U2R) recall: XGBoost 0.70 vs RF_v2 0.30 vs RF_v1 0.09
   - **Test Set**: Marginal improvement for minority classes
     - Class 4 (U2R) recall: XGBoost 0.10 vs RF_v2 0.08 vs RF_v1 0.00
   - **Technical Analysis**: XGBoost's boosting mechanism adapts to minority class examples but fails to generalize

3. **Prediction Distribution Analysis**:
   - **Over-prediction of Probe (Class 2)**: 4,375 predictions vs 2,421 actual examples
   - **Under-prediction of Normal (Class 0)**: 9,409 predictions vs 12,131 actual examples
   - **Technical Impact**: Lower precision for Probe class (0.37) compared to RF_v2 (0.70)
   - **Potential Cause**: XGBoost's focus on hard-to-classify examples shifts decision boundaries

4. **Feature Importance Insights**:
   - **Key Finding**: Service types (eco_i, ecr_i) dominated importance metrics
   - **Technical Implication**: Model heavily reliant on specific network services
   - **Comparison**: Different feature utilization than RF models, suggesting complementary information

5. **Algorithm-Specific Behavior**:
   - **Validation Superiority**: XGBoost excelled on validation (sequential optimization)
   - **Test Set Challenges**: Worse generalization than RF_v2 despite better validation metrics
   - **Technical Analysis**: Likely more susceptible to distribution shift between train and test
   - **Implementation Note**: Default learning rate potentially too aggressive for this domain

---

### **Conclusion**
This initial XGBoost implementation offers valuable insights into the application of gradient boosting for multiclass network intrusion detection. The model demonstrates compelling validation performance, particularly for minority classes, but exhibits significant generalization issues when applied to the external test set. Overall, XGBoost achieves 65% accuracy on the test set, underperforming the enhanced Random Forest model (RF_v2) at 76.1%, though slightly outperforming RF_v2 on minority class detection.

The contrasting behavior between validation and test performance highlights a critical challenge in network intrusion detection systems: while XGBoost's sequential optimization excels at fitting complex patterns in the training data, it appears more vulnerable to the distribution shift present in the NSL-KDD dataset. This suggests that boosting algorithms may require more careful regularization in security domains where test distributions can differ significantly from training.

A key technical advantage of the XGBoost implementation is its built-in feature importance analysis, which identified service-specific features (service_eco_i and service_ecr_i) as the most influential for classification. This insight differs from previous feature selection approaches and could inform future hybrid models.

The most surprising finding is XGBoost's exceptional performance on minority classes in the validation set (0.70 recall for U2R vs. 0.30 for RF_v2), which fails to translate to the test set, where improvement is marginal (0.10 vs. 0.08). This suggests that while boosting can effectively adapt to rare patterns during training, without explicit mechanisms to handle distribution shift, these gains don't generalize well.

Future work should explore:
1. Explicit regularization techniques (e.g., gamma, min_child_weight) to control overfitting
2. Learning rate schedules to better balance fitting and generalization
3. Integration of data balancing techniques (as in RF_v2) with XGBoost's inherent capabilities
4. Model stacking or ensemble approaches combining Random Forest and XGBoost strengths
5. Feature engineering guided by the distinct feature importance profiles of different algorithms

This implementation demonstrates that while XGBoost offers powerful capabilities for multiclass classification, it requires careful tuning and additional techniques to handle the severe distribution shifts present in network intrusion detection datasets. The complementary strengths of RF and XGBoost suggest that ensemble approaches combining both algorithms may yield superior overall performance.