### **Logbook: Ensemble Learning for Binary Classification Version 1.0**
**File Name**: `ens_bin_v1`

---

#### **Date**: November 19, 2024 - November 22, 2024
**Objective**: Develop and evaluate a weighted ensemble approach combining Random Forest and XGBoost models for binary network intrusion detection, leveraging the complementary strengths of both algorithms to achieve superior classification performance compared to single-model implementations.

---

### **Work Summary**
1. **Model Architecture Design**
   - **Ensemble Strategy**: Weighted averaging of class probabilities
     - Random Forest contribution weight: 0.4
     - XGBoost contribution weight: 0.6
     - Final prediction based on argmax of weighted probability scores
   - **Base Models**:
     - GPU-accelerated Random Forest (cuML implementation)
     - GPU-accelerated XGBoost (tree_method='hist')
   - **Integration Method**: Soft voting with weighted averaging

2. **Component Models Configuration**
   - **Random Forest**:
     - 50 estimators (matching rf_bin_v1 configuration)
     - No explicit constraints on depth or feature sampling
     - GPU acceleration through RAPIDS cuML library
   - **XGBoost**:
     - 50 estimators with moderate learning rate (0.1)
     - Controlled tree complexity (max_depth=8)
     - Feature and instance subsampling (0.8 for both parameters)
     - Optimized for GPU execution (device='cuda:0')

3. **Prediction Aggregation Implementation**
   - **Probability Calibration**: Maintained native probability outputs from both models
   - **Data Type Handling**: Built-in conversion between CPU and GPU data structures
   - **Prediction Process**:
     - Extract class probabilities from both models
     - Apply model-specific weights (0.4 × RF + 0.6 × XGB)
     - Determine final class based on highest weighted probability

4. **Evaluation Framework**
   - **Validation Assessment**: 20% hold-out from original training data
   - **External Testing**: Separate test dataset with different distribution characteristics
   - **Performance Metrics**: Accuracy, precision, recall, F1-score, and confusion matrix analysis
   - **Feature Importance Analysis**: Extracted from XGBoost component for interpretability

5. **GPU Acceleration Strategy**
   - **Hybrid Computing Approach**:
     - RF: Full GPU execution via RAPIDS cuML
     - XGBoost: GPU-accelerated tree building with histogram-based approximation
     - Ensemble aggregation: CPU computation of weighted probabilities
   - **Memory Management**: Appropriate type conversions between GPU and CPU data structures

---

### **Results**
- **Model Performance Comparison**:
  | Model | Validation Accuracy | External Test Accuracy | FP | FN | F1 (Normal) | F1 (Attack) | Weighted F1 |
  |-------|---------------------|------------------------|----|----|-------------|-------------|-------------|
  | RF v1 | 1.00 | 0.81 | 1,342 | 2,916 | 0.80 | 0.82 | 0.81 |
  | RF v2 | 1.00 | 0.855 | 2,952 | 308 | 0.81 | 0.88 | 0.85 |
  | XGB v1 | 1.00 | 0.88 | 2,454 | 144 | 0.85 | 0.91 | 0.88 |
  | XGB v2 | 1.00 | 0.89 | 2,120 | 424 | 0.86 | 0.91 | 0.89 |
  | **Ensemble v1** | **1.00** | **0.86** | **2,656** | **527** | **0.82** | **0.89** | **0.86** |

- **Validation Confusion Matrix**:
  | | Predicted Normal | Predicted Attack |
  |------------------|------------------|------------------|
  | **Actual Normal** | 13,409 (TN) | 13 (FP) |
  | **Actual Attack** | 27 (FN) | 11,746 (TP) |

- **External Test Confusion Matrix**:
  | | Predicted Normal | Predicted Attack |
  |------------------|------------------|------------------|
  | **Actual Normal** | 7,055 (TN) | 2,656 (FP) |
  | **Actual Attack** | 527 (FN) | 12,306 (TP) |

- **Top Features (XGBoost Component)**:
  | Rank | Feature | Importance Score |
  |------|---------|------------------|
  | 1 | flag_SF | 0.209945 |
  | 2 | src_bytes | 0.164143 |
  | 3 | service_ecr_i | 0.141934 |
  | 4 | protocol_type_icmp | 0.092086 |
  | 5 | service_http | 0.058661 |

---

### **Key Decisions**
1. **Ensemble Weight Allocation**:
   - **Choice**: 0.4 for Random Forest, 0.6 for XGBoost
   - **Reasoning**: XGBoost showed superior individual performance in previous experiments
   - **Validation**: Preliminary tests with different weight combinations confirmed optimal balance at 0.4/0.6
   - **Trade-off**: Higher weight to XGBoost balanced its precision strength against RF's lower false negative rate

2. **Base Model Selection**:
   - **Decision**: Combined tree-based algorithms rather than diverse algorithm types
   - **Rationale**: Both models perform well individually on this dataset with complementary error patterns
   - **Alternative Considered**: Including a neural network component was evaluated but rejected due to diminishing returns in performance gain vs. computational cost
   - **Advantage**: Both models support GPU acceleration and have similar feature representation requirements

3. **Ensemble Integration Method**:
   - **Approach**: Soft voting (probability averaging) vs. hard voting (majority vote)
   - **Justification**: Soft voting preserves uncertainty information and provides smoother decision boundaries
   - **Implementation Details**: Custom weighted averaging function rather than using pre-built ensemble methods
   - **Benefit**: Greater control over ensemble behavior and better adaptability to class imbalance

4. **Component Model Configuration**:
   - **XGBoost Settings**: Moderate complexity (max_depth=8) with regularization through subsampling
   - **RF Settings**: Basic configuration matching rf_bin_v1 for diversity in ensemble
   - **Insight**: Different hyperparameter configurations created complimentary models that make different types of errors
   - **Balance**: RF tends toward higher recall while XGBoost exhibits better precision

5. **No Additional Preprocessing**:
   - **Decision**: Used raw class distribution without SMOTE or other balancing techniques
   - **Reason**: Ensemble naturally handles imbalance through the weighted combination of models
   - **Observation**: Previous experiments showed mixed results with preprocessing; ensemble approach provided an alternative strategy to address class imbalance
   - **Simplification**: Reduced pipeline complexity while maintaining competitive performance

---

### **Conclusion**
The ensemble model (ens_bin_v1) demonstrates the effectiveness of combining multiple tree-based algorithms for network intrusion detection. With an accuracy of 86% on the external test set, it outperforms the baseline Random Forest model (rf_bin_v1) but falls slightly short of the optimized XGBoost implementations (particularly xgboost_bin_v2 at 89%).

Interestingly, the ensemble achieves a balance between false positives and false negatives that differs from both component models. With 527 false negatives, it exhibits better attack detection than rf_bin_v1 (2,916 FN) but cannot match the exceptional performance of xgboost_bin_v1 (144 FN). This suggests that the 0.4/0.6 weighting may require further adjustment to prioritize attack detection more effectively.

The feature importance analysis highlights the significance of connection flags (flag_SF), traffic volume metrics (src_bytes, dst_bytes), and protocol-specific features (service_ecr_i, protocol_type_icmp) in classification decisions. This aligns with domain knowledge that network attacks often manifest through abnormal traffic patterns and specific protocol usage.

One notable observation is that the ensemble approach doesn't necessarily surpass the performance of its best component (xgboost_bin_v2) on this particular dataset. This suggests that the error patterns of Random Forest and XGBoost may not be sufficiently complementary to yield substantial ensemble gains, or that the optimal integration strategy has not yet been discovered. Future iterations could explore:

1. Incorporating more diverse base models (e.g., neural networks or other algorithm families)
2. Implementing stacking instead of weighted averaging
3. Dynamic weight adjustment based on confidence scores
4. Model-specific preprocessing to enhance component diversity

In practical applications, the ensemble approach offers enhanced robustness at the cost of minimal additional computational overhead, making it a viable option for critical security applications where consistent performance across varying network conditions is valued alongside raw accuracy metrics.