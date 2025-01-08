# Import relevant libraries
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from keras.utils import pad_sequences
from tensorflow.keras.preprocessing import sequence

def getKmers(Sequence, size=3):
    return [Sequence[x:x+size].lower() for x in range(len(str(Sequence)) - size + 1)]
#df9['words']= df9.apply(lambda x: getKmers(x['Sequence']), axis=1)
#df9= df9.drop('Sequence', axis=1)

def extract_features(df, max_length=500):
    # Extract k-mers for different sizes
    df['kmer_6'] = df['Sequence'].apply(lambda x: getKmers(x, size=6))
    df['kmer_4'] = df['Sequence'].apply(lambda x: getKmers(x, size=4))
    #df['kmer_5'] = df['Sequence'].apply(lambda x: getKmers(x, size=4))
  
    # Drop original sequence column
    df = df.drop('Sequence', axis=1)

    # Tokenizer setup
    tokenizer = Tokenizer(char_level=True)

    # Extract features for each k-mer size
    X_3 = tokenizer.fit_on_texts(df['kmer_6'].values)
    X_3 = tokenizer.texts_to_sequences(df['kmer_6'].values)
    X_3 = pad_sequences(X_3, maxlen=max_length)

    X_4 = tokenizer.fit_on_texts(df['kmer_4'].values)
    X_4 = tokenizer.texts_to_sequences(df['kmer_4'].values)
    X_4 = pad_sequences(X_4, maxlen=max_length)

    #X_5 = tokenizer.fit_on_texts(df['kmer_5'].values)
    #X_5 = tokenizer.texts_to_sequences(df['kmer_5'].values)
    #X_5 = pad_sequences(X_5, maxlen=max_length)

    # Combine features
    X_combined = np.concatenate([X_4,X_5], axis=1)

    return X_combined
