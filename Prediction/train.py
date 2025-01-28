import numpy as np
import pandas as pd
# Import Tokenizer and pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from keras.utils import pad_sequences
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('enzyme_l4_upto2023.csv')
df1=df.iloc[:,[1,4,5]]
df1.rename(columns={'Entry': 'identifier','EC number':'class'}, inplace=True)
# Define a regular expression to match only complete four-digit EC numbers 
ec_pattern = r'^\d+\.\d+\.\d+\.\d+$'
# Filter the DataFrame to keep only rows with complete four-digit EC numbers
pivot_df = df1[df1['class'].str.match(ec_pattern, na=False)]
unique_classes_count = pivot_df['class'].nunique()
print(unique_classes_count)

def getKmers(Sequence, size=3):
    return [Sequence[x:x+size].lower() for x in range(len(str(Sequence)) - size + 1)]
#df9['words']= df9.apply(lambda x: getKmers(x['Sequence']), axis=1)
#df9= df9.drop('Sequence', axis=1)

def extract_features(df, max_length=500):
    # Extract k-mers for different sizes
    df['kmer_6'] = df['Sequence'].apply(lambda x: getKmers(x, size=6))
    df['kmer_4'] = df['Sequence'].apply(lambda x: getKmers(x, size=4))
    #df['kmer_5'] = df['Sequence'].apply(lambda x: getKmers(x, size=5))
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
    X_combined = np.concatenate([X_3,X_4], axis=1)

    return X_combined

# Apply feature extraction
X_combined = extract_features(pivot_df)
print(X_combined.shape)
y_data=pivot_df['class']
label_encoder = LabelEncoder()
# Fit and transform y_data
y_true = label_encoder.fit_transform(y_data)
# Compute the number of classes
ec_classes = [str(ec) for ec in le.classes_]  # Convert to string for display
n_classes = len(ec_classes)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_true, test_size = 0.10, shuffle=True)

# The value of gamma is set by getting the best accuracy at different values of sigma
gamma = 2.0
# Initialize sample weights
sample_weights = np.ones(len(y_train))

# Iterative reweighting
for iteration in range(3):  # Fixed number of iterations
    # Train Random Forest with current sample weights
    rf = RandomForestClassifier(n_estimators=100, random_state=42,class_weight='balanced',n_jobs=-1)
    gb_clf = lgb.LGBMClassifier(class_weight='balanced',n_jobs=1)
    dtree = DecisionTreeClassifier(class_weight='balanced',max_depth=10)
    # Create an ensemble classifier with soft voting optimized weightage
    ensemble_clf = VotingClassifier(estimators=[
       ('random_forest', rf_clf),
       ('gradient_boosting', gb_clf),
       ('decision_tree',dtree),
    ], voting='soft', , weights=[2, 2, 1]) 
    ensemble_clf.fit(X_train, y_train, sample_weight=sample_weights)

    # Predict probabilities for the training set
    y_pred_proba = ensemble_clf.predict_proba(X_train)

    # Calculate class-wise focal weights
    y_pred_proba_clipped = np.clip(y_pred_proba, 1e-6, 1 - 1e-6)
    focal_weights = np.zeros_like(sample_weights)

    for i, (true_class, proba) in enumerate(zip(y_train, y_pred_proba_clipped)):
        p_t = proba[true_class]
        focal_weights[i] = (1 - p_t) ** gamma

    # Update sample weights for the next iteration
    sample_weights = focal_weights

# Final evaluation on the test set
y_test_pred = ensemble_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_test_pred)
#scaler=MinMaxScaler(feature_range=(-1,1))
#X_train_scaled=scaler.fit_transform(X_train)
#X_test_scaled=scaler.fit_transform(X_test)
unique_classes_in_test = np.unique(y_test)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

y_prob = ensemble_clf.predict_proba(X_test)
# Get top 20 predictions
y_pred_top5 = np.argsort(y_prob, axis=1)[:, -20:]
# Calculate top-5 accuracy
top5_accuracy = top_k_accuracy_score(y_test, y_prob, k=5,labels=classes)
# Print the top-5 accuracy
print(f"Top-5 Accuracy: {top5_accuracy:.3f}")
top_n_values = range(1,20)  # calculate top-n accuracy for n=1 to n=20
accuracy_scores = [top_k_accuracy_score(y_test, y_prob, k=n,labels=classes) for n in top_n_values]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(top_n_values, accuracy_scores, marker='o')
plt.xlabel('Top-n')
plt.ylabel('Accuracy')
plt.title('Top-n Accuracy')
plt.grid(True)

# Printing and writing the results to a file
print(report)
print(f"Accuracy: {accuracy}\n")

# Writing to a file
with open('uniprot_unseen.txt', 'w') as file:
    file.write(report)
    file.write(f"\nAccuracy: {accuracy}\n")
    file.write(f"Macro Precision: {precision:.4f}\n")
    file.write(f"Macro Recall: {recall:.4f}\n")
    file.write(f"Macro F1-Score: {f1:.4f}\n")
    file.write(f"Accuracy: {accuracy:.4f}\n") 
