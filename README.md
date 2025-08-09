Income Prediction API with FastAPI

This project demonstrates a full ML pipeline for predicting income levels (≤50K or >50K) based on census data. The pipeline includes data preprocessing, model training, evaluation on data slices, RESTful API creation, and automated testing with GitHub Actions.

GitHub Link
https://github.com/seane13/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/actions/workflows/manual.yml

Project Overview

-  Dataset: Census data (downloaded separately)
-  Model: Random Forest Classifier (or other, depending on your implementation)
-  API: Built with FastAPI
-  Tests: Pytest unit tests
-  CI: GitHub Actions
-  Packaging: requirements.txt / environment.yml
---

 Directory Structure

├── data/
│   └── census.csv            # Raw dataset (tracked via DVC)
├── model/
│   ├── model.pkl             # Trained model
│   ├── encoder.pkl           # OneHotEncoder for categorical features
│   └── lb.pkl                # LabelBinarizer for labels
├── ml/
│   ├── data.py               # Data preprocessing logic
│   └── model.py              # ML training, inference, slice evaluation
├── tests/
│   └── test_ml.py            # Unit tests
├── main.py                   # FastAPI app
├── local_api.py              # Script to test the API
├── slice_output.txt          # Metrics on categorical slices
├── model_card.md             # Model documentation
├── requirements.txt          # Pip dependencies
├── environment.yml           # Conda environment definition
├── .github/workflows/main.yml  # CI pipeline
└── README.md


---

 Model Performance

Evaluation was performed using precision, recall, and F1-score across different slices of the data.
- Example (Workclass = Private):
  Precision: 0.7362
  Recall: 0.6384
  F1: 0.6838

Detailed performance across all slices is logged in slice_output.txt.
---

 API Usage


Start the API locally

uvicorn main:app --reload



Interact using local script

python local_api.py


- GET / returns a welcome message
- POST /data/ accepts JSON data and returns a prediction
pytest test_ml.py -v

Ensure flake8 passes as well:
flake8 .



 Continuous Integration



This project uses GitHub Actions to enforce code quality:

- Runs pytest and flake8 on every push
- Python version matches development environment




 License



MIT License

---



Acknowledgements



This project was developed as part of the Udacity Machine Learning DevOps Engineer Nanodegree.
