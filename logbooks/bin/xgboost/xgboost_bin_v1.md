### **Logbook: GPU-Accelerated XGBoost for Binary Classification Version 1.0**
**File Name**: `xgboost_bin_v1`

---

#### **Date**: November 11, 2024 - November 14, 2024
**Objective**: Implement and evaluate an XGBoost classifier for binary network intrusion detection on the NSL-KDD dataset, utilizing GPU acceleration to improve computational efficiency while maintaining high detection accuracy, and compare performance against previously developed Random Forest models.

---

### **Work Summary**
1. **Model Architecture Selection**
   - **Algorithm**: Gradient Boosting Decision Trees via XGBoost
   - **Implementation**: GPU-accelerated XGBClassifier
   - **Configuration**:
     - n_estimators: 50 (same as rf_bin_v1)
     - tree_method: 'hist' (histogram-based approximation for GPU efficiency)
     - device: 'cuda' (explicit GPU utilization)
     - Default values for other hyperparameters (learning_rate=0.3, max_depth=6)

2. **Data Processing Approach**
   - **Dataset**: Preprocessed NSL-KDD data (125,973 samples, 108 features)
   - **Label Distribution**: 67,343 normal (53.5%), 58,630 attack (46.5%)
   - **Training/Validation Split**: 80/20 ratio using stratified sampling
   - **Data Types**: Native cuDF dataframes for GPU compatibility
   - **Feature Selection**: Full feature set utilized without dimensionality reduction

3. **Feature Importance Analysis**
   - **Top Features** (by gain importance):
     1. service_ecr_i (0.374)
     2. src_bytes (0.196)
     3. service_http (0.125)
     4. diff_srv_rate (0.056)
     5. logged_in (0.032)
   - **Insight**: Protocol-specific features and traffic volume metrics dominate predictive power
   - **Comparison**: Different feature importance distribution than tree-based Random Forest models

4. **Evaluation Protocol**
   - **Internal Validation**: 20% stratified hold-out (25,194 samples)
   - **External Testing**: Separate test dataset (22,544 samples)
   - **Metrics**: Accuracy, precision, recall, F1-score, confusion matrix
   - **Implementation**: Combined cuML and sklearn metrics for comprehensive evaluation

---

### **Results**
- **Model Performance Comparison**:
  | Metric | XGBoost (v1) Internal | XGBoost (v1) External | RF (v1) External | RF (v2) External |
  |--------|------------------------|------------------------|-------------------|-------------------|
  | Accuracy | 1.00 | 0.88 | 0.81 | 0.855 |
  | F1-score (Class 0) | 1.00 | 0.85 | 0.80 | 0.81 |
  | F1-score (Class 1) | 1.00 | 0.91 | 0.82 | 0.88 |
  | Precision (Class 0) | 1.00 | 0.98 | 0.74 | 0.96 |
  | Recall (Class 0) | 1.00 | 0.75 | 0.86 | 0.70 |
  | Precision (Class 1) | 1.00 | 0.84 | 0.88 | 0.81 |
  | Recall (Class 1) | 1.00 | 0.99 | 0.77 | 0.98 |

- **Confusion Matrix (External Test)**:
  | | Predicted Normal | Predicted Attack | Total |
  |------------------|------------------|------------------|-------|
  | **Actual Normal** | 7,257 (TN) | 2,454 (FP) | 9,711 |
  | **Actual Attack** | 144 (FN) | 12,689 (TP) | 12,833 |
  | **Total** | 7,401 | 15,143 | 22,544 |

- **Feature Importance**:
  | Rank | Feature | Importance Score |
  |------|---------|------------------|
  | 1 | service_ecr_i | 0.374037 |
  | 2 | src_bytes | 0.195503 |
  | 3 | service_http | 0.125345 |
  | 4 | diff_srv_rate | 0.056305 |
  | 5 | logged_in | 0.032049 |

---

### **Key Decisions**
1. **Algorithm Selection**:
   - **Choice**: XGBoost over traditional GBDTs or Random Forests
   - **Rationale**: Combines gradient boosting's sequential error correction with GPU optimization
   - **Advantage**: Achieved higher accuracy (88% vs 85.5% for rf_bin_v2) with minimal hyperparameter tuning

2. **GPU Acceleration Method**:
   - **Approach**: Histogram-based tree construction ('hist' method)
   - **Tradeoff**: Slight approximation in split finding vs. significant speedup in training
   - **Benefit**: Enabled rapid model development without sacrificing performance

3. **Hyperparameter Configuration**:
   - **Strategy**: Used XGBoost defaults without extensive tuning unlike rf_bin_v2
   - **Decision Factor**: Initial performance exceeded optimized RF models
   - **Future Direction**: Potential for further gains through hyperparameter optimization

4. **Class Imbalance Handling**:
   - **Approach**: No explicit resampling techniques (unlike SMOTE in rf_bin_v2)
   - **Observation**: XGBoost inherently handled class imbalance effectively
   - **Result**: Achieved higher recall on minority class (0.99) than RF models even without resampling

5. **Feature Engineering**:
   - **Decision**: Utilized full feature set without feature selection
   - **Insight**: XGBoost's built-in regularization mitigated overfitting despite high dimensionality
   - **Outcome**: Strong generalization to external test data without explicit feature selection

---

### **Conclusion**
The XGBoost classifier implementation (xgboost_bin_v1) demonstrates superior performance for binary network intrusion detection compared to both Random Forest implementations. Without extensive hyperparameter tuning or advanced preprocessing techniques, it achieved an external test accuracy of 88%, outperforming both rf_bin_v1 (81%) and the optimized rf_bin_v2 (85.5%).

Most notably, XGBoost exhibits exceptional attack detection capability with a 99% recall rate on the attack class, while maintaining reasonable precision (84%). This represents a significant improvement over both RF models and suggests that the sequential, boosting-based learning approach more effectively captures the patterns distinguishing network attacks from normal traffic.

The feature importance analysis reveals that service-specific features (particularly ICMP echo reply service) and basic traffic metrics (src_bytes) contribute most significantly to classification decisions. This differs somewhat from RF models and provides complementary insights into the discriminative patterns within the dataset.

One particularly impressive aspect is XGBoost's effective handling of class imbalance without explicit resampling techniques. While rf_bin_v2 required SMOTE to achieve balanced performance, XGBoost inherently adapted to the class distribution, likely due to its gradient-based learning approach.

The false negative rate (144 missed attacks) is less than half that of rf_bin_v2 (308), representing a substantial security improvement, as missed attacks typically pose greater risk than false alarms. However, the model still generates a significant number of false positives (2,454), which, while better than rf_bin_v2 (2,952), suggests room for improvement in normal traffic recognition.

Future work should explore hyperparameter optimization techniques similar to those implemented in rf_bin_v2, including learning rate scheduling, regularization parameter tuning, and advanced sampling methods. Additionally, implementing feature selection based on the identified importance metrics could potentially further improve computational efficiency while maintaining or enhancing detection accuracy.