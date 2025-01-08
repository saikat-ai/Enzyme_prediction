# SOLVE
# Enzyme Function Prediction through subsequence tokenization and ensemble learning framework
1. In Datasets folder contains all the test-dataset for enzyme function prediction in different hierarchy labels
2. Codes folder contains three files 
- Feature_extraction.py: to extract subsequence based feature from any given protein sequence
- train.py: train an ensemble classifier and predict the EC number for a given protein sequence
# Usage
- go to the Prediction folder.
- download the model.joblib file from google drive
- Then run the mono-functional_EC.py code
- Paste the description of your sequence (for example: > seq12)
- Paste the input sequence
- Prediction of 4-digit Ec number will be shown.
# Requirements
- numpy = 1.19.5
- scikit-learn = 0.24.2
- tensorflow   = 2.6.2
- keras        = 2.6.0
