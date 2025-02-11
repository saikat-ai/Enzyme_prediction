##importing relevant libraries
import numpy as np 
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import joblib
# Import Tokenizer and pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from keras.utils import pad_sequences
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('enzyme_l4_upto2023.csv')
df1=df.iloc[:,[1,4,5]]
df1.rename(columns={'Entry': 'identifier','EC number':'class'}, inplace=True)
# Define a regular expression to match only complete four-digit EC numbers
ec_pattern = r'^\d+\.\d+\.\d+\.\d+$'
# Filter the DataFrame to keep only rows with complete four-digit EC numbers
pivot_df = df1[df1['class'].str.match(ec_pattern, na=False)]

# Initialize an empty dictionary for class mapping and a counter
class_dict = dict()
count = 1

# Get the value counts for the 'class' column in df6
classes = pivot_df['class'].value_counts().items()

# Create a new DataFrame to store filtered data
filtered_df = pivot_df.copy()

for cat, num in classes:
    # Remove all classes that have less than 100 values
    if num < 10:
        temp = pivot_df['class'] == cat
        filtered_df = filtered_df[~temp].copy()
    else:
        # Map remaining classes to unique numerical identifiers
        class_dict[cat] = count
        count += 1
unique_classes_count = filtered_df['class'].nunique()
print(unique_classes_count)

## Feature extraction from sequence
def getKmers(Sequence, size=6):
    return [Sequence[x:x+size].lower() for x in range(len(str(Sequence)) - size + 1)]

def extract_features(df, tokenizers=None, max_length=500):
    # Extract k-mers for different sizes
    df['kmer_3'] = df['Sequence'].apply(lambda x: getKmers(x, size=2))
    df['kmer_4'] = df['Sequence'].apply(lambda x: getKmers(x, size=4))
    df['kmer_5'] = df['Sequence'].apply(lambda x: getKmers(x, size=6))

    # Tokenizers initialization
    if tokenizers is None:
        tokenizer_3 = Tokenizer(char_level=True)
        tokenizer_4 = Tokenizer(char_level=True)
        tokenizer_5 = Tokenizer(char_level=True)

        # Fit tokenizers on the respective k-mer data
        tokenizer_3.fit_on_texts(df['kmer_3'].values)
        tokenizer_4.fit_on_texts(df['kmer_4'].values)
        tokenizer_5.fit_on_texts(df['kmer_5'].values)

        tokenizers = {
            'tokenizer_3': tokenizer_3,
            'tokenizer_4': tokenizer_4,
            'tokenizer_5': tokenizer_5
        }

    # Process k-mer 3
    X_3 = tokenizers['tokenizer_3'].texts_to_sequences(df['kmer_3'].values)
    X_3 = pad_sequences(X_3, maxlen=max_length)

    # Process k-mer 4
    X_4 = tokenizers['tokenizer_4'].texts_to_sequences(df['kmer_4'].values)
    X_4 = pad_sequences(X_4, maxlen=max_length)

    # Process k-mer 5
    X_5 = tokenizers['tokenizer_5'].texts_to_sequences(df['kmer_5'].values)
    X_5 = pad_sequences(X_5, maxlen=max_length)

    # Combine features
    X_combined = np.concatenate([X_3, X_4, X_5], axis=1)

    return X_combined, tokenizers
X_combined, tokenizers = extract_features(filtered_df, max_length=500)

#from sklearn import preprocessing
#min_max_scaler = preprocessing.MinMaxScaler(feature_range =(-1, 1))
y_data=filtered_df['class']
# Encode class labels
le = LabelEncoder()
y_encoded = le.fit_transform(y_data)
# Compute the number of classes
ec_classes = [str(ec) for ec in le.classes_]  # Convert to string for display
n_classes = len(ec_classes)

# Splitting the human dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_combined,
                                                    y_encoded,
                                                    test_size = 0.20,
                                                    shuffle=True,random_state=5)
gamma_values = [0.5, 1.0, 1.2, 1.5, 2.0, 2.5]
accuracies = []

best_overall_accuracy = 0
best_rf_model = None
accuracy_results = []
best_metrics = {
    "gamma": None,
    "iteration": None,
    "precision": 0,
    "recall": 0,
    "f1_score": 0
}

alpha = 4  # Scaling factor for dynamic lambda

for gamma in gamma_values:
    # Initialize sample weights
    sample_weights = np.ones(len(y_train))
    best_iteration_accuracy = 0
    best_iteration_rf = None
    best_iteration_metrics = {
        "iteration": None,
        "precision": 0,
        "recall": 0,
        "f1_score": 0
    }
    
    # Iterative reweighting
    for iteration in range(10):  # Fixed number of iterations
        # Train Random Forest with current sample weights
        rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, n_jobs=-1)
        rf.fit(X_train, y_train, sample_weight=sample_weights)

        # Predict probabilities and labels for the training set
        y_pred_proba = rf.predict_proba(X_train)
        y_pred = rf.predict(X_train)

        # Calculate class-wise focal weights
        y_pred_proba_clipped = np.clip(y_pred_proba, 1e-6, 1 - 1e-6)
        focal_weights = np.zeros_like(sample_weights)

        for i, (true_class, proba) in enumerate(zip(y_train, y_pred_proba_clipped)):
            num_classes = len(proba)
            if true_class >= num_classes or true_class < 0:
                    continue  # Skip if true class is out of range
            p_t = np.clip(proba[true_class], 1e-6, 1 - 1e-6)
            lambda_dynamic = 1 + alpha * (1 - p_t)  # Adaptive lambda
            focal_weights = (1 - p_t) ** gamma
            
            # Boost weight only for misclassified samples
            if true_class != y_pred[i]:
                focal_weights *= lambda_dynamic

        # Update sample weights for the next iteration
        sample_weights[i] = focal_weights

        # Evaluate model performance on test set
        y_test_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, average='weighted')
        recall = recall_score(y_test, y_test_pred, average='weighted')
        f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        # Track best iteration model for the current gamma
        if accuracy > best_iteration_accuracy:
            best_iteration_accuracy = accuracy
            best_iteration_rf = rf
            best_iteration_metrics = {
                "iteration": iteration,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }

    # Generate classification report
    #classification_rep = classification_report(y_test, best_iteration_rf.predict(X_test))
    
    # Store best model accuracy for current gamma
    accuracies.append(best_iteration_accuracy)
    accuracy_results.append((gamma, best_iteration_accuracy, best_iteration_metrics))
    print(f"Gamma: {gamma}, Best Accuracy: {best_iteration_accuracy:.4f}, Best Iteration: {best_iteration_metrics['iteration']}, Precision: {best_iteration_metrics['precision']:.4f}, Recall: {best_iteration_metrics['recall']:.4f}, F1 Score: {best_iteration_metrics['f1_score']:.4f}")
    
    # Save classification report to text file
    with open("accuracy_results.txt", "a") as f:
        f.write(f"Gamma: {gamma}\n")
        f.write(f"Best Iteration: {best_iteration_metrics['iteration']}\n")
        f.write(f"Accuracy: {best_iteration_accuracy:.4f}\n")
        f.write(f"Precision: {best_iteration_metrics['precision']:.4f}\n")
        f.write(f"Recall: {best_iteration_metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {best_iteration_metrics['f1_score']:.4f}\n")
        f.write("\n" + "-" * 50 + "\n")

# Save the best model
#joblib.dump(best_rf_model, "model.joblib")
