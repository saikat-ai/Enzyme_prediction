import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the saved model, tokenizer, and label encoder
tokenizer = joblib.load("tokenizer_final.joblib")  # Load tokenizer
model = joblib.load("model_final.joblib")  # Load trained model
label_encoder = joblib.load("label_encoder_final.joblib")  # Load LabelEncoder for EC number decoding

# Step 2: Define the k-mer extraction function
def getKmers(sequence, size=3):
    return [sequence[x:x+size].lower() for x in range(len(str(sequence)) - size + 1)]

# Step 3: Extract features from sequence
def extract_features(sequence, tokenizer, max_length=500):
    seq_df = pd.DataFrame({'Sequence': [sequence]})  # Convert to DataFrame

    # Extract k-mers for different sizes
    seq_df['kmer_3'] = seq_df['Sequence'].apply(lambda x: getKmers(x, size=2))
    seq_df['kmer_4'] = seq_df['Sequence'].apply(lambda x: getKmers(x, size=3))
    seq_df['kmer_5'] = seq_df['Sequence'].apply(lambda x: getKmers(x, size=6))

    # Process k-mer 3
    X_3 = tokenizer['tokenizer_3'].texts_to_sequences(seq_df['kmer_3'].values)
    X_3 = pad_sequences(X_3, maxlen=max_length)

    # Process k-mer 4
    X_4 = tokenizer['tokenizer_4'].texts_to_sequences(seq_df['kmer_4'].values)
    X_4 = pad_sequences(X_4, maxlen=max_length)

    # Process k-mer 5
    X_5 = tokenizer['tokenizer_5'].texts_to_sequences(seq_df['kmer_5'].values)
    X_5 = pad_sequences(X_5, maxlen=max_length)

    # Combine features
    X_combined = np.concatenate([X_3, X_4, X_5], axis=1)

    return X_combined

# Step 4: Read sequence from user input
def input_fasta_sequence():
    print("\nEnter the protein sequence in FASTA format (Only sequence, without headers):")
    sequence = input().strip().upper()  # Convert to uppercase
    return sequence

# Step 5: Predict Top 5 EC numbers with probabilities
def predict_top5_ec_numbers(sequence, top_n=5):
    X_input = extract_features(sequence, tokenizer)  # Extract features

    # Predict EC numbers (Single-label classification with probabilities)
    y_pred_prob = model.predict_proba(X_input)[0]  # Get probability scores for all classes

    # Get the indices of the top N probabilities
    top_ec_indices = np.argsort(y_pred_prob)[::-1][:top_n]  # Indices of highest probabilities

    # Map indices to actual EC numbers
    top_ec_numbers = label_encoder.inverse_transform(top_ec_indices)  # Get EC numbers
    top_ec_probs = y_pred_prob[top_ec_indices]  # Get corresponding probabilities

    # Format output with EC numbers and their actual probabilities
    predicted_ecs_with_probs = [f"{ec} ({prob:.3f})" for ec, prob in zip(top_ec_numbers, top_ec_probs)]

    return "; ".join(predicted_ecs_with_probs) if predicted_ecs_with_probs else "No confident predictions found."

# Step 6: Interactive loop for user input and EC prediction
def main():
    print("\n### Mono-Functional Enzyme EC Number Prediction ###")
    
    while True:
        sequence = input_fasta_sequence()  # Get user input sequence
        predicted_ecs = predict_top5_ec_numbers(sequence)  # Predict Top 5 EC numbers
        
        print("\nPredicted EC Numbers (Top 5):")
        print(predicted_ecs if predicted_ecs else "No confident predictions found.")

        # Ask if the user wants to continue
        cont = input("\nDo you want to input another sequence? (yes/no): ").strip().lower()
        if cont != "yes":
            print("Exiting...")
            break

# Run the script
if __name__ == "__main__":
    main()
