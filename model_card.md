# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a supervised binary classification model trained to predict whether an individual’s income exceeds $50K per year based on census data.
- **Model Type**: Random Forest Classifier
- **Framework**: scikit-learn 1.5.1
- **Deployment**: FastAPI 0.112.0
- **Random State**: 42 for reproducibility

## Intended Use
The model is intended for educational purposes to demonstrate:
- Training and evaluating a binary classification model
- Handling categorical data preprocessing
- Computing performance on demographic slices
- Serving an ML model via a REST API

## Training Data
The model was trained on the UCI Adult Income dataset (Census Income data).
Categorical features were one-hot encoded, and the label is binary: `>50K` or `<=50K`.

## Evaluation Data
The test data is a held-out portion of the original dataset, not seen during training. It reflects similar distributions to the training data and is used to evaluate generalization and subgroup fairness.

## Metrics
The model is evaluated using:
- Precision
- Recall
- F1-score (beta=1)

**Overall Performance**:
Precision: **0.7391**
Recall: **0.6384**
F1-score: **0.6851**

## Performance on Categorical Slices
Below are example F1-scores for selected subgroups (see `slice_output.txt` for the full list):

- **Paste your latest slice results here from `slice_output.txt`**
- Example format:
  - Education: Bachelors — n=105 | precision=0.7456 | recall=0.6123 | f1=0.6720
  - Education: HS-grad — n=200 | precision=0.5110 | recall=0.5200 | f1=0.5155
  - Sex: Male — n=567 | precision=0.7001 | recall=0.6900 | f1=0.6950

## Ethical Considerations
- **Bias/Fairness**: The model shows uneven performance across groups. Some slices may have lower recall and F1-scores, suggesting potential bias.
- **Privacy**: While the model uses public census data, deploying similar models with real user data may require privacy safeguards.
- **Transparency**: All code and processing steps are available in this repository.

## Caveats and Recommendations
- Not calibrated for probability outputs — decision thresholds should be validated before use.
- Performance varies across demographic groups; fairness-aware techniques may improve equity.
- Retraining is recommended if applied to new or shifted data distributions.

## Decision Threshold
The classifier uses the default probability threshold of 0.5.

## Reproducibility
1. Install dependencies from `requirements.txt` in a Python 3.10+ environment.
2. Run `python train_model.py` to process the data, train the model, and save artifacts.
3. View console output for overall metrics and `slice_output.txt` for slice metrics.
