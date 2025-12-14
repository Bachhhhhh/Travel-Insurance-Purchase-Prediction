# Travel Insurance Purchase Prediction

---

## Table of Contents

- [Project Overview](#-project-overview)
- [Requirements](#-requirements)
- [Workflow](#-workflow)
  - [Data Loading & EDA](#-data-loading-&-eda)
  - [Data Preprocessing](#-data-preprocessing)
  - [Feature Engineering](#-feature-engineering)
  - [Model Implementation](#-model-implementation)
    - [Class Imbalance Check](#-class-imbalance-check)
    - [Model 1: Decision Tree + Resampling](#-decision-tree-+-resampling)
    - [Model 2: Decision Tree + SMOTE](#-decision-tree-+-smote)
    - [Model 3: Random Forest](#-random-forest)
- [Performance Comparison & Evaluation](#-performance-comparison-&-evaluation)
- [Conclusion](#-conclusion)

---

## Project Overview

This project focuses on predicting whether a customer will purchase **travel insurance** based on demographic, financial, and travel-related features. 
The problem is formulated as a **binary classification task** and is solved using supervised machine learning models like **Decision Tree** and **Random Forest**.

The dataset contains attributes such as age, employment type, annual income, family size, health conditions, and travel history. 
A key challenge in this task is **class imbalance**, which is addressed using multiple sampling strategies.

---

## Requirements

The following Python libraries are required to run this project:

```txt
scikit-learn==1.3.2
scipy==1.11.4
imbalanced-learn==0.11.0
```

## Data Loading & EDA

The dataset used in this project is in this file:

```text
TravelInsurancePrediction.csv
```

```python
import pandas as pd

df = pd.read_csv("TravelInsurancePrediction.csv")
df.head()
```

---

## Data Preprocessing

The preprocessing step focuses on cleaning and preparing raw data before modeling.

**Main tasks include:**

* Dropping null or invalid data points
* Removing unnecessary columns
* Filling missing values if required

```python
def preprocess_data(df):
    """
    Preprocess data (e.g., drop null datapoints, unnecessary columns, or fill missing data)
    :param df: pandas DataFrame
    :return: pandas DataFrame
    """
    # Implementation here
    return df
```

---

## Feature Engineering

Feature engineering transforms categorical and boolean variables into numerical representations suitable for machine learning models.

**Key steps:**

* Encode categorical features
* Normalize numerical features
* Visualize feature correlation using a heatmap

```python
def feature_engineering(df):
    """
    Apply all feature engineering to transform data into numeric format
    :param df: pandas DataFrame
    :return: pandas DataFrame
    """
    return df
```

### Feature Matrix and Target

```python
def create_X_y(df):
    """
    Feature engineering and create X and y
    :param df: pandas DataFrame
    :return: X (DataFrame), y (Series)
    """
    return X, y
```

### Train-Test Split

```python
from sklearn.model_selection import train_test_split

trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.3, random_state=42)
```

---

## Implements Machine Learning Models

### Class Imbalance Check

Training label distribution:

```text
{0: 894, 1: 496}
```

The dataset is **imbalanced**, with class `0` being the majority. Therefore, resampling techniques are applied.

---

## Model 1: Decision Tree + Resampling (sklearn.utils)

### Upsampling Function

```python
from sklearn.utils import resample

def upsampling_func(X, y):
    # Upsample minority class
    return X_resampled, y_resampled
```

### Model Pipeline

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from imblearn import FunctionSampler


def build_model_1(X, y):
    upsampler = FunctionSampler(func=upsampling_func)
    clf = DecisionTreeClassifier(random_state=42)

    pipe = Pipeline(steps=[
        ('upsampling', upsampler),
        ('tree', clf)
    ])

    param_grid = {
        'tree__criterion': ['gini', 'entropy', 'log_loss'],
        'tree__max_depth': [None, 5, 10, 20],
        'tree__min_samples_split': [2, 5, 10],
        'tree__min_samples_leaf': [1, 2, 4],
        'tree__max_features': [None, 'sqrt', 'log2'],
        'tree__class_weight': [None, 'balanced']
    }

    model = GridSearchCV(pipe, param_grid, scoring='f1', cv=5, n_jobs=2)
    model.fit(X, y)
    return model
```

### Performance

* Precision: **0.946**
* Recall: **0.570**
* Accuracy: **0.834**
* F1-score: **0.711**

Best parameters:

```text
{'tree__criterion': 'gini', 'tree__max_depth': 5, 'tree__min_samples_leaf': 4}
```

Feature importance and decision tree visualization are shown below:

> *Feature importance plot*
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/67f20d62-1cdf-4ad2-9379-f1e15acaac4f" />

> *Decision tree visualization*
<img width="1570" height="790" alt="image" src="https://github.com/user-attachments/assets/c0faa8a8-136e-4489-8793-dce65007b9e1" />


---

## Model 2: Decision Tree + SMOTE

SMOTE is applied to synthetically oversample the minority class.

### Performance

* Precision: **0.767**
* Recall: **0.617**
* Accuracy: **0.796**
* F1-score: **0.684**

Best parameters:

```text
{'tree__criterion': 'entropy', 'tree__max_depth': 5, 'tree__min_samples_leaf': 2}
```

> *Feature importance with SMOTE*
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/03c7fe1a-c8b1-4c8e-9754-2fadf1fc0c0f" />

> *Decision tree visualization (SMOTE)*
<img width="1570" height="790" alt="image" src="https://github.com/user-attachments/assets/e8440c05-00bf-4712-bf9f-c3d7a6b2905c" />

---

## Model 3: Random Forest

Random Forest is trained with `class_weight='balanced'` to address class imbalance.

```python
from sklearn.ensemble import RandomForestClassifier

def build_model_3(X, y):
    rf = RandomForestClassifier(random_state=42)
    pipe = Pipeline(steps=[('rf', rf)])

    param_grid = {
        'rf__n_estimators': [20, 50, 75],
        'rf__criterion': ['gini', 'entropy'],
        'rf__max_depth': [None, 5, 10, 20],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4],
        'rf__max_features': [None, 'sqrt', 'log2'],
        'rf__class_weight': ['balanced']
    }

    model = GridSearchCV(pipe, param_grid, scoring='recall', cv=5, n_jobs=8)
    model.fit(X, y)
    return model
```

### Performance

* Precision: **0.725**
* Recall: **0.654**
* Accuracy: **0.787**
* F1-score: **0.688**

Execution time: **452.66 seconds**

> *Random Forest feature importance*
<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/c939d4eb-7e17-465f-aca1-e60bf1e889cb" />

> *Sample decision tree from Random Forest*
<img width="1570" height="812" alt="image" src="https://github.com/user-attachments/assets/c0592ec7-993d-444d-9069-acd57924f549" />


---

## Performance Comparison & Evaluation

| Model                    | Precision | Recall | Accuracy | F1-score |
| ------------------------ | --------- | ------ | -------- | -------- |
| Decision Tree + Resample | 0.946     | 0.570  | 0.834    | 0.711    |
| Decision Tree + SMOTE    | 0.767     | 0.617  | 0.796    | 0.684    |
| Random Forest            | 0.725     | 0.654  | 0.787    | 0.688    |

Based on the comparison table, the Decision Tree model with Resampling achieved the highest Precision (0.946) and Accuracy (0.834), while Random Forest achieved the highest Recall (0.654) 
with a balanced F1-score (0.688). Decision Tree with SMOTE shows a slightly lower F1-score (0.684) but improves Recall (0.617) compared to simple resampling. 
In general, if the goal is to detect customers who will buy insurance (maximize Recall), Random Forest is the most suitable choice. 
However, if the goal is to minimize false positives (maximize Precision), Decision Tree with Resampling should be prioritized. In this insurance prediction task, Recall is selected as the main evaluation metric 
because missing a potential customer (False Negative) leads to direct revenue loss. Therefore, Random Forest is chosen as the final model due to its superior Recall performance.

---

## Conclusion

Through this project, I gained practical experience in handling **imbalanced classification problems**, applying **different sampling techniques**, and comparing multiple machine learning models. 
I learned how to design robust **ML pipelines**, tune hyperparameters using **GridSearchCV**, and evaluate models based on business-oriented metrics such as Recall. 
This project strengthened my understanding of how model selection should align with real-world objectives rather than relying solely on accuracy.
