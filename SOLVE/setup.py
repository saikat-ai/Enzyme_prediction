# setup.py
from setuptools import setup, find_packages

setup(
    name="SOLVE",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "tensorflow",
        "pandas"
    ],
    author="Saikat",
    author_email="cssd2399@iacs.res.in",
    description="A library for extracting k-mer features from protein sequences",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourgithub/SOLVE",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

# src/solve/__init__.py
"""
SOLVE - A package for extracting k-mer features from protein sequences.
"""
from .feature_extraction import extract_features

# src/solve/feature_extraction.py
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_kmers(sequence, size=3):
    return [sequence[x:x+size].lower() for x in range(len(str(sequence)) - size + 1)]

def extract_features(df, max_length=500):
    df['kmer_6'] = df['Sequence'].apply(lambda x: get_kmers(x, size=6))
    df['kmer_4'] = df['Sequence'].apply(lambda x: get_kmers(x, size=4))
    #df['kmer_5'] = df['Sequence'].apply(lambda x: get_kmers(x, size=2))
    
    df = df.drop('Sequence', axis=1)
    
    tokenizer = Tokenizer(char_level=True)
    
    X_6 = tokenizer.fit_on_texts(df['kmer_6'].values)
    X_6 = tokenizer.texts_to_sequences(df['kmer_6'].values)
    X_6 = pad_sequences(X_6, maxlen=max_length)
    
    X_4 = tokenizer.fit_on_texts(df['kmer_4'].values)
    X_4 = tokenizer.texts_to_sequences(df['kmer_4'].values)
    X_4 = pad_sequences(X_4, maxlen=max_length)
    
    #X_5 = tokenizer.fit_on_texts(df['kmer_5'].values)
    #X_5 = tokenizer.texts_to_sequences(df['kmer_5'].values)
    #X_5 = pad_sequences(X_5, maxlen=max_length)
    
    X_combined = np.concatenate([X_6, X_4], axis=1)
    
    return X_combined
