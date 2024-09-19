# Titanic Survival Prediction - Machine Learning Models Comparison

This repository contains an end-to-end solution for predicting the survival of passengers on the Titanic using various machine learning algorithms. The goal of this project is to apply and evaluate a variety of classification models, both individual and ensemble, to determine the best model for predicting passenger survival. The dataset used for this project is from the [Kaggle Titanic competition](https://www.kaggle.com/c/titanic).

You can also check out my Kaggle profile here: [Fares Ashraf's Kaggle Profile](https://www.kaggle.com/faresashraf1001).

## Project Overview

In this project, we explore and implement multiple machine learning techniques on the Titanic dataset to predict whether a passenger survived or not based on their characteristics (age, gender, class, etc.). The following steps were covered:

1. **Data Preprocessing**: Handling missing values, feature engineering, and encoding categorical variables.
2. **Modeling**: Applying a variety of machine learning models, including both individual classifiers and ensemble methods.
3. **Model Selection**: Using `GridSearchCV` for hyperparameter tuning and cross-validation to find the best models.
4. **Evaluation**: Comparing the performance of different models based on accuracy and classification reports, and selecting the best-performing model.

## Models Applied

### Single Models:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Support Vector Machine (SVM)
- Decision Tree

### Ensemble Models:
- Random Forest
- Extra Trees Classifier
- Bagging Classifier
- AdaBoost
- Gradient Boosting
- XGBoost
- Voting Classifier (Soft Voting)
- Stacking Classifier

## Hyperparameter Tuning

Each model underwent hyperparameter tuning using `GridSearchCV` to find the optimal parameters for the best performance. The cross-validation strategy used is Stratified K-Folds, ensuring that class imbalance is taken into account.

## Results

The models were evaluated based on their accuracy and classification reports (precision, recall, F1-score) on the test set. After testing various models, the **Voting Classifier** with soft voting yielded the best performance, combining the strengths of Random Forest and XGBoost.

## Conclusion

This project demonstrates the effectiveness of using ensemble methods to boost performance in classification problems. The Voting Classifier, which combines the predictions of multiple well-performing models, achieved the best accuracy in this competition.

## Next Steps

- Further feature engineering to extract more meaningful insights.
- Experiment with deep learning models for better performance.
- Investigate different ways of handling missing data to improve model generalization.
