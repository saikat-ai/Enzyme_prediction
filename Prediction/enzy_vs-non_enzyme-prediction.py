import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

def getKmers(sequence, size=3):
    return [sequence[x:x+size].lower() for x in range(len(str(sequence)) - size + 1)]

tokenizer = joblib.load("tokenizer5.joblib") 
model = joblib.load('model5.joblib')
label_encoder = joblib.load('label_encoder5.joblib')

def extract_features(sequence, tokenizer, max_length=500):
    seq_df = pd.DataFrame({'Sequence': [sequence]})

    # Extract k-mers for different sizes
    seq_df['kmer_3'] = seq_df['Sequence'].apply(lambda x: getKmers(x, size=2))
    seq_df['kmer_4'] = seq_df['Sequence'].apply(lambda x: getKmers(x, size=4))
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

def interactive_session():
    while True:
        print("\nEnter the protein sequence (Only sequence, without headers):")
        sequence = input().strip().upper()
        
        # Extract features
        features = extract_features(sequence, tokenizer)
        
        # Predict
        prediction = model.predict(features)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        
        print(f"Predicted Class: {predicted_label}")
        
        # Ask if the user wants to continue
        cont = input("Do you want to input another sequence? (yes/no): ").strip().lower()
        if cont != 'yes':
            print("Exiting...")
            break

if __name__ == "__main__":
    interactive_session()
