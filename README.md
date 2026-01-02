# Predicting 30-Day Hospital Readmission Using Machine Learning

## Overview
Hospital readmissions within 30 days are a major concern in healthcare due to increased costs and poorer patient outcomes. This project applies and compares multiple machine learning classification models to predict 30-day hospital readmission using the **Diabetes 130-US Hospitals dataset**.

We evaluate model performance under **severe class imbalance** and demonstrate why relying solely on accuracy can be misleading in clinical settings. Advanced evaluation metrics such as **ROC-AUC, recall, precision, and Brier score** are used to guide model selection.

---

## Dataset
- **Source:** Diabetes 130-US Hospitals dataset (UCI Machine Learning Repository)
- **Samples:** 101,766 hospital encounters
- **Features:** 49 original features (demographics, diagnoses, medications, lab results)
- **Target Variable:**
  - Binary classification
  - `1`: Readmitted within 30 days
  - `0`: Not readmitted within 30 days
- **Class Distribution:**
  - Readmitted (<30 days): ~11%
  - Not readmitted: ~89%

---

## Data Preprocessing
- Dropped ID-like and high-missing-value features:
  - `encounter_id`, `patient_nbr`, `weight`, `payer_code`, `medical_specialty`
- Replaced `"?"` values with `NaN`
- Created a binary target variable from the original `readmitted` column
- Applied preprocessing pipelines using `ColumnTransformer`:
  - **Numerical features:** Mean imputation + standard scaling
  - **Categorical features:** Mode imputation + one-hot encoding

---

## Models Implemented
The following classifiers were implemented and compared:

- **k-Nearest Neighbors (kNN)**
- **Logistic Regression**
- **Multilayer Perceptron (MLP)**
- **XGBoost**

All models were implemented using **scikit-learn pipelines**, ensuring consistent preprocessing during training and evaluation.

---

## Experimental Setup
- **Data Split:**
  - Training: 60%
  - Validation: 20%
  - Test: 20%
  - Stratified splits to preserve class imbalance
- **Hyperparameter Tuning:**
  - `RandomizedSearchCV`
  - 3-fold cross-validation
  - Optimized initially for accuracy, then for recall
- **Reproducibility:** Fixed random seed (`random_state=178`)

---

## Evaluation Metrics
Given the imbalanced nature of the dataset, multiple metrics were used:
- Accuracy
- Precision & Recall
- ROC-AUC
- Brier Score (probability calibration)
- Confusion Matrices
- Learning Curves
- Jaccard Similarity (error overlap across models)

---

## Key Results

| Model      | Accuracy | ROC-AUC | Brier Score |
|-----------|----------|----------|-------------|
| kNN       | ~86%     | ~0.60    | ~0.11       |
| Logistic  | ~87%     | ~0.63    | ~0.10       |
| MLP       | ~89%     | ~0.68    | ~0.09       |
| XGBoost  | ~90%     | ~0.69    | ~0.085      |

### Insights
- Accuracy alone was misleading due to the 89/11 class imbalance
- All models initially learned to predict the majority class, leading to many false negatives
- Optimizing for **recall** significantly reduced false negatives, which is more appropriate for clinical use
- **XGBoost** achieved the best balance of recall, precision, and calibration
- Learning curves showed that complex models benefited most from increased data, while kNN did not scale well

---

## Visualizations
The project includes:
- Class distribution plots
- Missing value heatmaps
- Learning curves
- Confusion matrices
- Calibration plots
- Jaccard similarity analysis

These visualizations help explain model behavior beyond raw performance metrics.

---

## Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Project Structure
├── cs178_final_project.ipynb
├── report.pdf
├── README.md


---

## Contributors
- **Mohammadarshya Salehibakhsh**
  - Visualization, analysis, and report writing
- **Vishok Lakshmankumar**
  - Model training, evaluation metrics, and analysis
- **Eric Tao**
  - Recall-optimized training strategies and advanced evaluation

---

## Key Takeaways
This project highlights the importance of:
- Proper evaluation metrics for imbalanced datasets
- Model calibration in healthcare applications
- Experimental rigor and reproducibility
- Understanding trade-offs between simplicity, interpretability, and performance

---

## Future Work
- Incorporate cost-sensitive learning
- Explore temporal patient data
- Apply ensemble methods focused on recall optimization
- Evaluate clinical utility with domain-specific constraints
