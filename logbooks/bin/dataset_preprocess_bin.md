### **Logbook: NSL-KDD Dataset Preprocessing for Binary Classification**  
**File Name**: `dataset_preprocess_bin`  

---

#### **Date**: October 28, 2024 - October 31, 2024
**Objective**: Implement comprehensive preprocessing pipeline for the NSL-KDD dataset to prepare data for binary classification (normal vs. attack), including feature scaling, encoding, and selection to optimize model performance.

---

### **Work Summary**  
1. **Data Loading & Initial Analysis**  
   - **Source Data**: 
     - Training dataset: `KDDTrain+.csv` (125,973 samples, 42 features)
     - Testing dataset: `KDDTest+.csv` (22,544 samples, 42 features)
   - **Initial Inspection**:
     - Mixed data types (numeric and categorical)
     - Categorical features: protocol_type (3 values), service (70 values), flag (11 values)
     - Target transformation: Multi-class labels consolidated to binary (normal vs. anomaly)

2. **Feature Transformation**  
   - **Scaling**: 
     - Applied StandardScaler to all numeric features
     - Normalized numeric features to zero mean and unit variance
   - **Categorical Encoding**:
     - One-hot encoded categorical variables (protocol_type, service, flag)
     - Expanded feature space from 42 to 123 dimensions
     - Used integer encoding for dummy variables for memory efficiency

3. **Feature Selection Implementation**  
   - **Method 1: SelectKBest with Mutual Information**:
     - Applied information-theoretic approach to measure feature relevance
     - Configured to select top 20 most discriminative features
     - Key selected features included traffic-based and host-based attributes
   - **Method 2: Correlation-Based Feature Selection (CFS)**:
     - Implemented custom algorithm to balance relevance and redundancy
     - Eliminated features with correlation >0.8 to reduce multicollinearity
     - Removed constant features with zero variance
     - Selected 108 features based on target correlation and feature-feature interactions

4. **Preprocessing Pipeline Integration**  
   - **Training Data Processing**:
     - Sequential application of scaling → encoding → selection
     - Created binary target labels (0: normal, 1: anomaly)
     - Applied CFS for final feature set determination
   - **Testing Data Alignment**:
     - Applied identical transformation pipeline to test data
     - Ensured feature consistency by adding missing columns (with zeros)
     - Verified dimensional consistency between train and test sets

5. **Output Preparation**  
   - **Processed Datasets**:
     - Saved transformed train data (125,973 samples, 109 features) as CSV
     - Saved transformed test data (22,544 samples, 109 features) as CSV
   - **Preprocessing Objects**:
     - Serialized preprocessing components (scaler, selector, feature lists)
     - Saved objects using pickle for reproducible transformation application

---

### **Key Decisions**  
1. **Feature Scaling Approach**:  
   - Selected StandardScaler over MinMaxScaler or RobustScaler
   - **Rationale**: Network traffic features have extreme outliers; standardization reduces their impact while preserving feature distributions better than min-max scaling.

2. **Categorical Encoding Strategy**:  
   - Used one-hot encoding instead of ordinal encoding
   - **Trade-off**: Increased dimensionality (123 features) vs. accurate representation of categorical relationships
   - **Justification**: No inherent ordinality exists in protocol types or services; one-hot prevents introducing artificial relationships.

3. **Feature Selection Method**:  
   - Implemented two-stage selection (SelectKBest followed by CFS)
   - **Reasoning**: 
     - Information-theoretic metrics capture non-linear relationships better than correlation alone
     - CFS reduces redundancy while maintaining feature relevance
     - Hybrid approach leverages strengths of both filter methods

4. **Binary Label Creation**:  
   - All attack types mapped to single "anomaly" class (1)
   - **Consideration**: Loss of attack type specificity vs. simplified detection problem
   - **Decision factor**: Primary goal is anomaly detection, not attack classification

5. **Feature Alignment Strategy**:  
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
  | After Feature Selection (CFS)  | 125,973 × 109      | 22,544 × 109     |
  | Binary Label Distribution      | Not explicitly reported | Not explicitly reported |

- **Top Selected Features by Mutual Information**:
  - Traffic-based: src_bytes, dst_bytes, logged_in, count
  - Connection-based: serror_rate, srv_serror_rate, same_srv_rate
  - Host-based: dst_host_count, dst_host_srv_count, dst_host_diff_srv_rate
  - Service-specific: service_http, service_private
  - Flag-related: flag_S0, flag_SF

- **Processing Artifacts**:
  - Standardization scaler (fitted on training data)
  - Feature selector (SelectKBest, k=20)
  - Selected feature lists (CFS: 108 features)
  - Preprocessing pipeline serialized to disk

---

### **Conclusion**  
The preprocessing pipeline successfully transformed the NSL-KDD dataset into a format optimized for binary network intrusion detection. Starting with 42 raw features, the pipeline expanded the feature space through one-hot encoding and then applied sophisticated feature selection techniques to retain 109 informative attributes.

The correlation-based feature selection effectively balanced feature relevance with redundancy elimination, resulting in a dataset that preserves discriminative power while reducing dimensionality from the expanded feature space. The preprocessing maintained consistency between training and testing sets, ensuring valid model evaluation.

Key technical challenges addressed included handling the high cardinality of categorical features (particularly the 'service' field with 70 unique values), managing feature interactions, and ensuring proper alignment between train and test datasets. The implemented pipeline is modular and serialized, enabling consistent application to new data.

Future work could explore more advanced dimensionality reduction techniques like PCA or t-SNE, investigate class imbalance mitigation strategies, and implement feature engineering to create more discriminative attributes based on domain knowledge of network intrusion patterns. Additionally, evaluating the impact of different preprocessing choices on model performance would provide valuable insights for optimization.