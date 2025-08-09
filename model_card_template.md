# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a supervised binary classification model trained to predict whether an individual’s income exceeds $50K per year based on census data. It uses a Random Forest Classifier, trained on processed categorical and continuous features. The model is part of a pipeline built with scikit-learn, and is deployed via a FastAPI application.

## Intended Use
The model is intended for educational purposes to demonstrate how to:

- Train and evaluate a binary classification model
- Handle categorical data preprocessing
- Compute performance on demographic slices
- Serve an ML model via a RESTful API

## Training Data
The model was trained on the UCI Adult Income dataset, also known as “Census Income” data. The training set consists of demographic and employment-related features such as:

- Age
- Workclass
- Education
- Occupation
- Marital Status
- Race
- Sex
- Native Country
- Hours per week
- Capital gain/loss


Categorical features were one-hot encoded, and the label is binary: >50K or <=50K.

## Evaluation Data
The test data is a held-out portion of the original dataset, not seen during training. It reflects similar distributions to the training data and is used to evaluate generalization and subgroup fairness.

## Metrics
The model is evaluated using:

- Precision
- Recall
- F1-score


Overall Performance:

- Precision: 0.74
- Recall: 0.66
- F1-score: 0.69


Slice-based Evaluation (selected examples):

- Workclass: Private — F1: 0.6838
- Education: HS-grad — F1: 0.5114
- Marital-status: Never-married — F1: 0.5605
- Sex: Male — F1: 0.6985, Female — F1: 0.5995
- Race: White — F1: 0.6832, Black — F1: 0.6723


See slice_output.txt for full breakdown.

## Ethical Considerations
- Bias/Fairness: The model shows uneven performance across groups. Certain slices, such as 7th-8th education or Widowed, have low recall and F1-scores. These disparities suggest the model may encode or reinforce societal biases.
- Privacy: The model is trained on publicly available data. However, if extended to real-world settings, privacy concerns must be addressed.
- Transparency: This model is fully open, and both code and data are accessible for inspection and replication.

## Caveats and Recommendations
- The model is not suitable for high-stakes decision-making (e.g., hiring, credit).
- Performance varies across demographic groups. Further work is needed to address fairness.
- The model is not calibrated for probabilities. Use thresholds cautiously.
- Consider retraining the model with fairness constraints or techniques like reweighing or adversarial debiasing if intended for sensitive applications.
