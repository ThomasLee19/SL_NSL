# **Logbook: NSL-KDD Dataset Preprocessing for Multiclass Classification**
**File Name**: `dataset_preprocess_multi`

---

#### **Date**: November 23, 2024 - November 25, 2024
**Objective**: Implement comprehensive preprocessing pipeline for the NSL-KDD dataset to prepare data for multiclass classification of network intrusions, including feature scaling, encoding, and selection to optimize model performance across five distinct attack categories.

---

### **Work Summary**
1. **Data Loading & Initial Analysis**
   - **Source Data**:
     - Training dataset: `KDDTrain+.csv` (125,973 samples, 42 features)
     - Testing dataset: `KDDTest+.csv` (22,544 samples, 42 features)
   - **Initial Inspection**:
     - Mixed data types (numeric and categorical)
     - Categorical features: protocol_type (3 values), service (70 values), flag (11 values)
     - Target transformation: Attack types categorized into 5 distinct classes

2. **Attack Classification Schema**
   - **Multiclass Mapping Implementation**:
     - Class 0: Normal Traffic
     - Class 1: DOS (Denial of Service) - 6 attack types
     - Class 2: Probe (Surveillance/Scanning) - 6 attack types
     - Class 3: R2L (Remote to Local) - 8 attack types
     - Class 4: U2R (User to Root) - 7 attack types
   - **Class Distribution Analysis**:
     - Highly imbalanced (Class 4/U2R represents only 0.04% of training data)
     - Class imbalance patterns differ between train and test sets

3. **Feature Transformation**
   - **Scaling**:
     - Applied StandardScaler to all numeric features
     - Normalized numeric features to zero mean and unit variance
   - **Categorical Encoding**:
     - One-hot encoded categorical variables (protocol_type, service, flag)
     - Expanded feature space from 42 to 123 dimensions
     - Used integer encoding for dummy variables

4. **Feature Selection Implementation**
   - **Method 1: SelectKBest with Mutual Information**:
     - Applied information-theoretic approach to measure feature relevance
     - Configured to select top 20 most discriminative features
     - Selected features included src_bytes, dst_bytes, logged_in, and service_http
   - **Method 2: Correlation-Based Feature Selection (CFS)**:
     - Implemented custom algorithm to balance relevance and redundancy
     - Eliminated features with correlation >0.8 to reduce multicollinearity
     - Removed constant columns with zero variance
     - Selected 109 features based on target correlation and feature-feature interactions

5. **Preprocessing Pipeline Integration**
   - **Training Data Processing**:
     - Sequential application of scaling → encoding → multiclass labeling → selection
     - Created 5-class target labels based on attack type groupings
     - Applied CFS for final feature set determination
   - **Testing Data Alignment**:
     - Applied identical transformation pipeline to test data
     - Ensured feature consistency by adding missing columns (with zeros)
     - Verified dimensional consistency between train and test sets

6. **Output Preparation**
   - **Processed Datasets**:
     - Saved transformed train data (125,973 samples, 110 features) as CSV
     - Saved transformed test data (22,544 samples, 110 features) as CSV
   - **Preprocessing Objects**:
     - Serialized preprocessing components (scaler, selector, feature lists)
     - Saved objects using pickle for reproducible transformation application
     - Preserved attack mapping and class name dictionaries for reference

---

### **Key Decisions**
1. **Multiclass Approach Selection**:
   - Implemented 5-class categorization instead of binary classification
   - **Rationale**: More granular attack classification provides deeper insights into attack patterns and enables targeted defensive strategies.

2. **Feature Scaling Approach**:
   - Selected StandardScaler over alternatives
   - **Rationale**: Standardization effectively handles network traffic features with extreme outliers while preserving relative distributions.

3. **Categorical Encoding Strategy**:
   - Used one-hot encoding for categorical features
   - **Trade-off**: Increased dimensionality (123 features) vs. accurate representation of categorical relationships
   - **Justification**: Preserves non-ordinal nature of protocol types, services, and flags without introducing artificial relationships.

4. **Feature Selection Method**:
   - Implemented two-stage selection (SelectKBest followed by CFS)
   - **Reasoning**:
     - Information-theoretic metrics capture non-linear relationships better than correlation alone
     - CFS reduces redundancy while maintaining feature relevance
     - Hybrid approach balances feature relevance with dimensionality reduction

5. **Class Imbalance Handling**:
   - Preserved natural class distribution without resampling
   - **Consideration**: Extreme imbalance particularly in Class 4 (U2R) with only 52 samples
   - **Decision factor**: Maintaining realistic representation of attack frequency for model training

6. **Test Set Alignment Strategy**:
   - Added zero-filled columns for features present in training but missing in test
   - **Alternative considered**: Limiting features to intersection of train/test
   - **Selected approach**: Preserve all training features to maintain model compatibility

---

### **Results**
- **Preprocessed Dataset Properties**:
  | Aspect                         | Training Set       | Testing Set       |
  |--------------------------------|--------------------| -----------------|
  | Original Dimensions            | 125,973 × 42       | 22,544 × 42      |
  | After One-Hot Encoding         | 125,973 × 123      | 22,544 × 123     |
  | After Feature Selection (CFS)  | 125,973 × 110      | 22,544 × 110     |

- **Class Distribution**:
  | Class                           | Training Set          | Testing Set          |
  |---------------------------------|----------------------|----------------------|
  | Class 0 (Normal Traffic)        | 67,343 (53.46%)      | 9,711 (43.08%)       |
  | Class 1 (DOS)                   | 45,927 (36.46%)      | 5,741 (25.47%)       |
  | Class 2 (Probe)                 | 11,656 (9.25%)       | 2,421 (10.74%)       |
  | Class 3 (R2L)                   | 995 (0.79%)          | 2,199 (9.75%)        |
  | Class 4 (U2R)                   | 52 (0.04%)           | 52 (0.23%)           |

- **Top Selected Features by Mutual Information**:
  - Traffic-based: src_bytes, dst_bytes, logged_in, count, srv_count
  - Connection-based: serror_rate, srv_serror_rate, same_srv_rate, diff_srv_rate
  - Host-based: dst_host_count, dst_host_srv_count, dst_host_same_srv_rate
  - Service-specific: service_http
  - Flag-related: flag_S0, flag_SF

- **Processing Artifacts**:
  - Standardization scaler (fitted on training data)
  - Feature selector (SelectKBest, k=20)
  - Selected feature lists (CFS: 109 features)
  - Attack mapping dictionary (27 attack types to 5 classes)
  - Class names dictionary for reference
  - Preprocessing pipeline serialized for reproducibility

---

### **Conclusion**
The preprocessing pipeline successfully transformed the NSL-KDD dataset into a format optimized for multiclass network intrusion detection across five distinct attack categories. Starting with 42 raw features, the pipeline expanded the feature space through one-hot encoding and then applied sophisticated feature selection techniques to retain 110 informative attributes.

The correlation-based feature selection effectively balanced feature relevance with redundancy elimination, resulting in a dataset that preserves discriminative power while reducing dimensionality from the expanded feature space. The preprocessing maintained consistency between training and testing sets, ensuring valid model evaluation.

A notable challenge encountered was the extreme class imbalance, particularly for U2R attacks representing only 0.04% of training data. This imbalance suggests that special consideration will be needed during model training, potentially through class weighting, specialized loss functions, or ensemble approaches.

The difference in class distribution between training and testing sets (particularly for R2L attacks: 0.79% in training vs. 9.75% in testing) highlights a potential dataset shift that models will need to handle. This discrepancy suggests that performance on the test set may differ significantly from training validation metrics.

Future work could explore class balancing techniques, investigation of ensemble methods specialized for multiclass imbalanced data, and the development of hierarchical classification approaches that might better handle the natural grouping of attack types. Additionally, evaluating the impact of different preprocessing choices on model performance across each attack category would provide valuable insights for optimization.