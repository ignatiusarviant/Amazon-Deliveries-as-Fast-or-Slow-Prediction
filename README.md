# Classifying Risk of Amazon Delivery Delays Using Random Forest 

This project predicts whether an electronic delivery using a van in a metropolitan area is fast or not. We utilise machine learning to make this prediction, based on weather, traffic, distance, and other delivery-related factors.

## Overview

- The project filters the original dataset to focus only on electronic deliveries using vans in metropolitan areas.
- It calculates the geodesic distance between the store and the delivery location.
- The target variable is `fast_or_not`, which is 0 for faster than the median delivery time, and 1 for slower.
- The model used is a Random Forest Classifier.
- The ROC curve is visualised with performance metrics like accuracy, F1 score, recall, and AUC.

## About This Project

This project was created as part of my machine learning portfolio, focusing on logistic optimisation and classification. 
It demonstrates my skills in data cleaning, geospatial feature engineering, and model evaluation using a Random Forest Classifier.

## Folder Structure

- `data_processing.py`  
  Cleans the raw dataset, filters relevant rows, calculates distance, and creates a label (`fast_or_not`).

- `delivery_time_ml.py`  
  Trains a Random Forest model, evaluates it using various metrics, and plots a custom ROC curve.

- `Data Training & Test.csv`  
  Preprocessed dataset used for training and testing the machine learning model.

- `amazon_delivery.csv`  
  Original dataset (you can download it from Kaggle here:  
  https://www.kaggle.com/code/sujalsuthar/predicting-amazon-delivery-time-using-regression)

## How to Use

### 1. Install the Required Libraries
---
You can install the required libraries by running:

```bash
pip install pandas matplotlib seaborn scikit-learn geopy
```

### 2. Preprocess the Data
---
`python data_processing.py`

This will read amazon_delivery.csv, clean and transform the data, and export it as Data Training & Test.csv.

### 3. Preprocess the Data
---
`python delivery_time_ml.py`

This will train the Random Forest model and display the ROC curve, along with evaluation metrics, on the plot.

### 4. Evaluation Metrics
---
- F1 Score
- Accuracy
- Recall
- ROC-AUC Score
- OOB Score (Out-of-Bag score from Random Forest)

### Notes
---
- Only rows with Vehicle = van, Category = Electronics, and Area = Metropolitian are used.
- Rows with missing or zero coordinates are removed.
- The label fast_or_not is zero if delivery time is below the median, and one otherwise.
