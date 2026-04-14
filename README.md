# ADS508-final-project
Detect potentially fraudulent credit card transactions by analyzing patterns in historical transaction data. The goal of this project is to use machine learning techniques to identify suspicious transactions and help reduce financial fraud.

## Project Overview
In these datasets, fraudulent transactions account for only about ~0.15% of the total data. The goal of this project was to build a robust model that minimizes financial loss by catching fraud (High Recall) while maintaining a low false-positive rate.

## Technical Stack
* **Infrastructure:** AWS S3, Boto3
* **Data Processing:** Pandas, NumPy, Scikit-learn
* **Visualization:** Matplotlib, Seaborn
* **Handling Imbalance:** SMOTE (Synthetic Minority Over-sampling Technique)
* **Models:** Logistic Regression, Random Forest, XGBoost

## Pipeline Workflow

### 1. Data Ingestion & EDA
* Data is pulled directly from an AWS S3 bucket using boto3.
* Performed descriptive statistics and distribution analysis via Histograms and Boxplots.
* **Multicollinearity Check:** Used Variance Inflation Factor (VIF) to identify highly correlated features, leading to the log-transformation of transaction amounts.

### 2. Feature Engineering & Pre-processing
* **Log Transformation:** Applied log1p to the Amount field to reduce skewness and stabilize variance.
* **Scaling:** Standardized Time and log_amount using StandardScaler.
* **Resampling:** Utilized SMOTE on the training set to balance the Class distribution, ensuring the model learns the characteristics of fraudulent transactions rather than just the majority class.

### 3. Model Performance
We compared three models, evaluating them primarily on **ROC-AUC** and **Recall**:

| Model | Accuracy | Recall (Fraud) | ROC-AUC |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | 97.7% | 89% | 0.9727 |
| **Random Forest** | 99.9% | 83% | 0.9808 |
| **XGBoost** | 99.9% | 82% | 0.9754 |

## Key Insights
* **Feature Importance:** Latent features V17, V14, and V12 emerged as the strongest predictors across both Random Forest and XGBoost.
* **Precision-Recall Trade-off:** While Logistic Regression had the highest recall (89%), the ensemble models (RF/XGB) significantly reduced false positives, providing a more balanced precision-recall curve.

## Future Improvements
* **Real-time Ingestion:** Transition from batch processing to streaming data for proactive detection.
* **Automated Hyperparameter Tuning:** Implement SageMaker Hyperparameter Tuning jobs to optimize XGBoost's learning_rate and max_depth.
* **Drift Monitoring:** Use SageMaker Model Monitor to detect when the distribution of incoming transaction data shifts away from the training baseline.

---
