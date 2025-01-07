import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 2: Define the k-mer extraction function and feature extraction process
def getKmers(sequence, size=3):
    return [sequence[x:x+size].lower() for x in range(len(str(sequence)) - size + 1)]

def extract_features(df, tokenizers, max_length=500):
    # Extract k-mers for different sizes
    df['kmer_3'] = df['Sequence'].apply(lambda x: getKmers(x, size=2))
    df['kmer_4'] = df['Sequence'].apply(lambda x: getKmers(x, size=3))
    #df['kmer_5'] = df['Sequence'].apply(lambda x: getKmers(x, size=5))

    # Process k-mer 3
    X_3 = tokenizers['tokenizer_3'].texts_to_sequences(df['kmer_3'].values)
    X_3 = pad_sequences(X_3, maxlen=max_length)

    # Process k-mer 4
    X_4 = tokenizers['tokenizer_4'].texts_to_sequences(df['kmer_4'].values)
    X_4 = pad_sequences(X_4, maxlen=max_length)

    # Process k-mer 5
    #X_5 = tokenizers['tokenizer_5'].texts_to_sequences(df['kmer_5'].values)
    #X_5 = pad_sequences(X_5, maxlen=max_length)

    # Combine features
    X_combined = np.concatenate([X_3, X_4], axis=1)

    return X_combined

# Step 3: Load the tokenizers and the trained model
tokenizers = joblib.load('tokenizers1.joblib')
model = joblib.load('model.joblib')

# Step 4: Manually input sequences in FASTA format
def input_fasta():
    print("Enter protein sequences in FASTA format. Enter an empty line to finish input.")
    sequences = []
    while True:
        # Prompt user for sequence description (e.g., >seq1, >seq2)
        description = input("Enter sequence description (or press Enter to stop): ")
        if description == "":
            break
        # Prompt for the sequence itself
        sequence = input("Enter the sequence: ")
        sequences.append(sequence)
    return sequences

# Step 5: Get input sequences from the user
sequences = input_fasta()

# Step 6: Convert sequences into DataFrame
new_data = pd.DataFrame({'Sequence': sequences})

# Step 7: Extract features from the new dataset using the saved tokenizers
X_new = extract_features(new_data, tokenizers=tokenizers, max_length=500)

# Step 8: Make predictions
y_pred = model.predict(X_new)

# Step 9: Add predictions to the DataFrame
new_data['Prediction'] = y_pred

# Step 10: Display the results
print(new_data[['Sequence', 'Prediction']])

