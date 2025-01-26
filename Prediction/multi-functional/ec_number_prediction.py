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

# Step 5: Predict EC numbers
def predict_ec_numbers(sequence):
    X_input = extract_features(sequence, tokenizers)  # Extract features

    # Predict EC numbers (Multi-label classification)
    y_pred_prob = model.predict(X_input)
    y_pred_labels = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary labels

    # Map predictions back to EC numbers
    predicted_ecs = mlb.inverse_transform(y_pred_labels)

    return predicted_ecs

# Step 6: Interactive loop for user input and EC prediction
def main():
    print("\n### Enzyme EC Number Prediction ###")
    
    while True:
        sequence = input_fasta_sequence()  # Get user input sequence
        predicted_ecs = predict_ec_numbers(sequence)  # Predict EC numbers
        
        print("\nPredicted EC Numbers:")
        for ec_numbers in predicted_ecs:
            print("; ".join(ec_numbers))  # Display predicted EC numbers
        
        # Ask if the user wants to continue
        cont = input("\nDo you want to input another sequence? (yes/no): ").strip().lower()
        if cont != "yes":
            print("Exiting...")
            break

# Run the script
if __name__ == "__main__":
    main()
