# SOLVE
# Enzyme Function Prediction through subsequence tokenization and ensemble learning framework
1. In Datasets folder contains all the test-dataset for enzyme function prediction in different hierarchy labels
2. Codes folder contains three files 
- Feature_extraction.py: to extract subsequence based feature from any given protein sequence
- train.py: train an ensemble classifier and predict the EC number for a given protein sequence
- mono-functional_EC.py: Prediction for any unknown protein sequence
- # Usage
- go to the Prediction folder.
- download the model.joblib file from google drive
- Then run the enzyme_ec_numbers_with_proba.py (for promiscuous enzyme) or top-5_ec.py (for mono-funcational enzyme) script
- When promted, enter the protein sequence in Fasta format (only sequence without any header).the script will process the input sequence and display the top-5 ec numbers along with their probability scores.
- # Example:
[1] Enter the protein sequence in FASTA format (Only sequence, without headers):  
    MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVVTVEAAETFSLAAG 
    
[2] Predicted EC Numbers:  
    3.4.21.68 (0.34); 3.4.21.4 (0.32); 3.4.21.92 (0.38); 3.4.21.7 (0.28); 1.2.3.8 (0.074)
    
[3] Do you want to input another sequence? (yes/no):  
    if yes enter another sequence in same way and if no the program will be ended.
# Requirements
- numpy = 1.19.5
- scikit-learn = 0.24.2
- tensorflow   = 2.6.2
- keras        = 2.6.0
