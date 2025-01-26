import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Bio import SeqIO

# Step 1: Load the saved model and tokenizers
tokenizers = joblib.load("tokenizer_ensemble.joblib")  # Load tokenizer
model = joblib.load("model_multi.joblib")  # Load trained model
mlb = joblib.load("mlb_ensemble.joblib")  # Load MultiLabelBinarizer for EC number decoding

# Step 2: Define the k-mer extraction function
def getKmers(sequence, size=3):
    return [sequence[x:x+size].lower() for x in range(len(str(sequence)) - size + 1)]

# Step 3: Extract features from sequence
def extract_features(sequence, tokenizers, max_length=300):
    seq_df = pd.DataFrame({'Sequence': [sequence]})  # Convert to DataFrame

    # Extract k-mers for different sizes
    seq_df['kmer_3'] = seq_df['Sequence'].apply(lambda x: getKmers(x, size=2))
    seq_df['kmer_4'] = seq_df['Sequence'].apply(lambda x: getKmers(x, size=3))
    seq_df['kmer_5'] = seq_df['Sequence'].apply(lambda x: getKmers(x, size=6))

    # Process k-mer 3
    X_3 = tokenizers['tokenizer_3'].texts_to_sequences(seq_df['kmer_3'].values)
    X_3 = pad_sequences(X_3, maxlen=max_length)

    # Process k-mer 4
    X_4 = tokenizers['tokenizer_4'].texts_to_sequences(seq_df['kmer_4'].values)
    X_4 = pad_sequences(X_4, maxlen=max_length)

    # Process k-mer 5
    X_5 = tokenizers['tokenizer_5'].texts_to_sequences(seq_df['kmer_5'].values)
    X_5 = pad_sequences(X_5, maxlen=max_length)

    # Combine features
    X_combined = np.concatenate([X_3, X_4, X_5], axis=1)

    return X_combined

# Step 4: Read sequence from user input
def input_fasta_sequence():
    print("\nEnter the protein sequence in FASTA format (Only sequence, without headers):")
    sequence = input().strip().upper()  # Convert to uppercase
    return sequence

# Step 5: Predict EC numbers (Top 5 ECs only, with actual probabilities)
def predict_ec_numbers(sequence, top_n=5):
    X_input = extract_features(sequence, tokenizers)  # Extract features

    # Predict EC numbers (Multi-label classification)
    y_pred_prob = model.predict_proba(X_input)[0]  # Get probabilities for the sequence

    # Get the indices of the top N probabilities
    top_ec_indices = np.argsort(y_pred_prob)[::-1][:top_n]  # Indices of highest probabilities

    # Extract the actual EC numbers using the indices directly
    top_ec_numbers = mlb.classes_[top_ec_indices]  # Get EC numbers directly from the indices
    top_ec_probs = y_pred_prob[top_ec_indices]  # Get corresponding probabilities

    # Format the output with EC numbers and their actual probabilities
    predicted_ecs_with_probs = [f"{ec} ({prob:.3f})" for ec, prob in zip(top_ec_numbers, top_ec_probs)]

    return "; ".join(predicted_ecs_with_probs) if predicted_ecs_with_probs else "No confident predictions found."

# Step 6: Interactive loop for user input and EC prediction
def main():
    print("\n### Multi-Functional Enzyme EC Number Prediction ###")
    
    while True:
        sequence = input_fasta_sequence()  # Get user input sequence
        predicted_ecs = predict_ec_numbers(sequence)  # Predict EC numbers
        
        print("\nPredicted EC Numbers:")
        print(predicted_ecs if predicted_ecs else "No confident predictions found.")

        # Ask if the user wants to continue
        cont = input("\nDo you want to input another sequence? (yes/no): ").strip().lower()
        if cont != "yes":
            print("Exiting...")
            break

# Run the script
if __name__ == "__main__":
    main()
