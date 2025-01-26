# solve/tokenizer.py
import joblib
from tensorflow.keras.preprocessing.text import Tokenizer

class TokenizerLoader:
    def __init__(self, tokenizer_path='/content/drive/MyDrive/enzyme_train/mono_ec/tokenizer_ensemble.joblib'):
        self.tokenizer = joblib.load(tokenizer_path)
    
    def tokenize(self, texts):
        return self.tokenizer.texts_to_sequences(texts)
