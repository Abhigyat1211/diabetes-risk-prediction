# Diabetes Risk Prediction — Cost-Sensitive Clinical ML Pipeline

## Problem

Most diabetes-prediction projects optimize for accuracy and stop there. On this
dataset that's a trap: only ~8.5% of patients are diabetic, so a model that
predicts "no diabetes" for everyone already scores ~91% accuracy while being
clinically useless.

The two error types also aren't equally costly in a screening context:

- **False negative** (a diabetic patient flagged as healthy) → a missed
  diagnosis — a real health risk.
- **False positive** (a healthy patient flagged as at-risk) → an unnecessary
  follow-up test — a smaller, lower-cost inconvenience.

This project uses a weighted cost function (`5 × false negatives + 1 × false
positives`) as the primary evaluation metric throughout, rather than accuracy,
and treats every modeling decision — algorithm choice, hyperparameters, and
decision threshold — as something to optimize against that cost, not against
accuracy or ROC-AUC alone.

The dataset is a [diabetes prediction dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
of 100,000 patient records (gender, age, BMI, HbA1c level, blood glucose
level, hypertension, heart disease, and smoking history), used for binary
classification: predict whether a patient has diabetes.

## Approach

1. **EDA.** Explored relationships between blood glucose, HbA1c, BMI, and age
   against the diabetes label using joint plots and correlation analysis to
   identify the strongest candidate predictors and check for multicollinearity.
2. **Feature engineering.** Added a `bmi_age` interaction term, applied target
   encoding to `smoking_history`, and one-hot encoded `gender`, all inside a
   single `ColumnTransformer`-based preprocessing pipeline to keep training
   and inference consistent.
3. **Model comparison.** Benchmarked four classifiers — Logistic Regression,
   KNN, Random Forest, and XGBoost — on identical train/validation/test splits
   (stratified, 64/16/20).
4. **Cost-sensitive tuning.** Defined a custom cost function penalizing false
   negatives 5× more than false positives, wrapped it as a scorer, and used it
   — not accuracy or default ROC-AUC — to drive `RandomizedSearchCV` for
   Random Forest and XGBoost.
5. **Threshold optimization.** Swept the decision threshold on validation data
   to find the cost-minimizing cutoff for each tuned model, rather than
   defaulting to 0.5.
6. **Cross-validation.** Ran 5-fold stratified CV on the final XGBoost
   configuration to confirm the cost and AUC figures were stable across folds,
   not an artifact of one split.
7. **Interpretability.** Used SHAP values and XGBoost's gain-based feature
   importance to check which features were actually driving predictions.

## Results

| Model                         | Threshold | Cost (test) | ROC-AUC (test) |
| ------------------------------ | --------- | ----------- | --------------- |
| Logistic Regression (baseline) | 0.50      | 3,071       | 0.9601          |
| KNN (baseline)                 | 0.50      | 3,438       | 0.9358          |
| Random Forest (baseline)       | 0.50      | 2,858       | 0.9697          |
| XGBoost (baseline)             | 0.50      | 2,457       | 0.9741          |
| **Random Forest (tuned)**      | **0.1853**| **2,414**   | **0.9670**      |
| **XGBoost (tuned)**            | **0.5878**| **2,199**   | **0.9768**      |

**5-fold stratified CV (tuned XGBoost):** mean cost 2,120.4 ± 58.4, mean
ROC-AUC 0.9783 ± 0.0013 — confirming the result generalizes rather than
overfitting to one split.

Final XGBoost classification report (test set, tuned threshold):

| Class          | Precision | Recall | F1-score |
| -------------- | --------- | ------ | -------- |
| No diabetes    | 0.99      | 0.95   | 0.97     |
| Diabetes       | 0.61      | 0.85   | 0.71     |

The precision/recall trade-off on the diabetes class is intentional: catching
85% of true positive cases is prioritized over minimizing false alarms,
consistent with the 5:1 cost asymmetry defined above.

## Key finding

Threshold and hyperparameter tuning against the **cost function** — rather
than accuracy or default-threshold ROC-AUC — reduced total misclassification
cost by roughly **10.5%** for XGBoost (2,457 → 2,199) without materially
harming ROC-AUC (0.9741 → 0.9768). SHAP and feature-importance analysis
confirmed that **HbA1c level and blood glucose level** are, unsurprisingly,
the dominant predictors — together accounting for over 70% of the model's
gain-based feature importance — with the engineered `bmi_age` interaction as
a distant third.

## Repository contents

- `Diabetes_detection.ipynb` — full analysis, in order: EDA → feature
  engineering → preprocessing pipeline → baseline model comparison → custom
  cost function → hyperparameter tuning → threshold optimization → cross-
  validation → feature importance and SHAP interpretability.
- `diabetes_xgb_pipeline.joblib` — serialized, trained XGBoost pipeline
  (preprocessing + model) for direct inference.

## Tools

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, category_encoders,
seaborn, matplotlib.

## Running locally

```bash
pip install -r requirements.txt
jupyter notebook Diabetes_detection.ipynb
```

To use the trained model directly instead of re-running the notebook:

```python
import joblib
pipeline = joblib.load("diabetes_xgb_pipeline.joblib")
predictions = pipeline.predict(X_new)
```

`X_new` must contain the same columns as the training data (`gender`, `age`,
`hypertension`, `heart_disease`, `smoking_history`, `bmi`, `HbA1c_level`,
`blood_glucose_level`, `bmi_age`) — the `bmi_age` feature must be computed as
`bmi * age` before calling `.predict()`, since that transform happens
upstream of the saved pipeline, not inside it.
