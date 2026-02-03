# Personality Classification Project

## Overview
This project aims to classify individuals into **Introvert** or **Extrovert** personality types based on behavioral and social features. The dataset contains information such as time spent alone, social event attendance, friends circle size, and social media post frequency.

Machine learning models, including ensemble methods, Logistic Regression, SVM, and KNN, are used to predict personality type with high accuracy.

---

## Dataset

- **Source:** `personality_dataset.csv`
- **Number of entries:** 2,900
- **Features:**
  - `Time_spent_Alone`: Hours spent alone per day (numeric)
  - `Stage_fear`: Whether the person has stage fear (Yes/No)
  - `Social_event_attendance`: Frequency of attending social events (numeric)
  - `Going_outside`: Frequency of going outside (numeric)
  - `Drained_after_socializing`: Feeling drained after social interactions (Yes/No)
  - `Friends_circle_size`: Number of close friends (numeric)
  - `Post_frequency`: Frequency of social media posts (numeric)
  - `Personality`: Target variable (Introvert/Extrovert)

---

## Data Cleaning and Preprocessing

1. Removed duplicates.
2. Filled missing values:
   - Numeric features: imputed with random values based on personality type distributions.
   - Categorical features: imputed with mode values.
3. Converted categorical variables to numeric:
   - `Yes → 1`, `No → 0`
   - `Introvert → 1`, `Extrovert → 0`
4. Scaled features using `StandardScaler`.

---

## Exploratory Data Analysis (EDA)

- Visualized relationships between features and personality using bar plots and heatmaps.
- **Observations:**
  - More time alone → likely introvert
  - Stage fright → likely introvert
  - Higher social event attendance → likely extrovert
  - More frequent going outside → likely extrovert
  - Feeling drained after socializing → likely introvert
  - Larger friend circles → likely extrovert
  - Higher social media posting → likely extrovert
- Correlation heatmap showed relationships among numeric features.

---

## Models Implemented

- Random Forest Classifier
- Decision Tree Classifier
- AdaBoost Classifier
- Gradient Boosting Classifier
- Logistic Regression
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)

### Train vs Test Accuracy (before hyperparameter tuning):

| Model            | Train Accuracy | Test Accuracy |
|-----------------|----------------|---------------|
| AdaBoost         | 0.917          | 0.932         |
| SVC              | 0.926          | 0.932         |
| Gradient Boosting| 0.927          | 0.932         |
| Logistic Regression | 0.920       | 0.930         |
| KNN              | 0.929          | 0.922         |
| Random Forest    | 0.979          | 0.907         |
| Decision Tree    | 0.979          | 0.857         |

---

## Hyperparameter Tuning

Performed `GridSearchCV` with K-Fold Cross Validation (`k=8`).

| Model              | Best Params                                                         | Best Accuracy |
|------------------|--------------------------------------------------------------------|---------------|
| Random Forest     | `{'max_depth': 8, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}` | 0.926         |
| Decision Tree     | `{'criterion': 'entropy', 'max_depth': 8, 'min_samples_split': 10}` | 0.922         |
| AdaBoost          | `{'learning_rate': 0.01, 'n_estimators': 100}`                     | 0.926         |
| Gradient Boosting | `{'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 100}`     | 0.926         |
| Logistic Regression | `{'C': 0.01, 'penalty': 'l1', 'solver': 'liblinear'}`            | 0.925         |
| SVM               | `{'C': 0.1, 'gamma': 'scale', 'kernel': 'rbf'}`                    | 0.926         |
| KNN               | `{'metric': 'manhattan', 'n_neighbors': 9, 'weights': 'uniform'}`  | 0.924         |

---

## Best Model Performance

- **Model:** Gradient Boosting Classifier
- **Parameters:** `n_estimators=100`, `learning_rate=0.01`, `max_depth=5`, `random_state=42`
- **Metrics on Test Set:**
  - Accuracy: 0.93
  - Precision: 0.92
  - Recall: 0.92
  - F1 Score: 0.92

**Classification Report:**

|              | Precision | Recall | F1-score | Support |
|--------------|-----------|--------|----------|---------|
| Extrovert (0)| 0.94      | 0.94   | 0.94     | 282     |
| Introvert (1)| 0.92      | 0.92   | 0.92     | 221     |
| **Accuracy** |           |        | 0.93     | 503     |
| Macro avg    | 0.93      | 0.93   | 0.93     | 503     |
| Weighted avg | 0.93      | 0.93   | 0.93     | 503     |

- **ROC AUC Curve:** Visualized for model performance evaluation.

---

## Libraries Used

- `numpy`, `pandas`, `matplotlib`, `seaborn`, `plotly.express`
- `scikit-learn` (preprocessing, models, metrics, hyperparameter tuning)
- `xgboost` (optional for gradient boosting)

---

## Conclusion

- Ensemble methods (AdaBoost, Gradient Boosting) and SVC performed the best.
- Features such as time spent alone, friends circle size, and social media activity are strong indicators of personality.
- **Gradient Boosting** achieved 93% accuracy on the test set and is recommended for deployment.
