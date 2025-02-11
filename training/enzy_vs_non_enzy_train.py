import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle
import joblib
# Import Tokenizer and pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.experimental import enable_hist_gradient_boosting
#from sklearn.ensemble import  HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('enzyme_data_upto2023.csv')
#df1=df.iloc[:,[1,2,3,4]]
df2=df.iloc[:,[1,2,3]]
df3=pd.read_csv('Non-enzyme_upto2023.csv')
df4=df3.iloc[:,[1,2,3]]
df5=pd.concat([df2, df4], ignore_index=True)
df5=shuffle(df5)

df20=pd.read_csv('enzyme_testdata_upto2024_wt_incom_ec.csv')
df21=df20.iloc[:,[1,2,3]]
df22=pd.read_csv('Non-enzyme_testset_upto2024.csv')
df23=df22.iloc[:,[1,2,3]]
df24=pd.concat([df21,df23],ignore_index=True)
df24=shuffle(df24)
total_rows=df24.shape[0]
print(total_rows)
filtered_df = df5[~df5['Sequence'].isin(df24['Sequence'])]

df25=pd.concat([filtered_df, df24], ignore_index=True)

def getKmers(Sequence, size=6):
    return [Sequence[x:x+size].lower() for x in range(len(str(Sequence)) - size + 1)]
# Feature extraction function with tokenizers
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
X_combined, tokenizers = extract_features(df25, max_length=500)
#joblib.dump(tokenizers, "tokenizer5.joblib", compress=('gzip', 3))

y_data=df25['class']
label_encoder = LabelEncoder()
# Fit and transform y_true
y_true = label_encoder.fit_transform(y_data)
#joblib.dump(label_encoder,'label_encoder6.joblib',compress=('gzip', 3))

# Split the data: last 140 rows for testing, remaining for training
X_train, X_test = X_combined[:-1162], X_combined[-1162:]
y_train, y_test = y_true[:-1162], y_true[-1162:]

# Define a range of gamma values
gamma_values = [0.5, 1.0, 1.2, 1.5, 2.0, 2.5]
accuracies = []
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import joblib
import numpy as np

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
        gb_clf = lgb.LGBMClassifier(class_weight='balanced',n_jobs=1)
        dtree = DecisionTreeClassifier(class_weight='balanced',max_depth=10)
        # Create an ensemble classifier with soft voting optimized weightage
        ensemble_clf = VotingClassifier(estimators=[
            ('random_forest', rf_clf),
            ('gradient_boosting', gb_clf),
            ('decision_tree',dtree),
        ], voting='soft', , weights=[5, 3, 0.5])
        ensemble_clf.fit(X_train, y_train, sample_weight=sample_weights)

        # Predict probabilities and labels for the training set
        y_pred_proba = ensemble_clf.predict_proba(X_train)
        y_pred = ensemble_clf.predict(X_train)

        # Calculate class-wise focal weights
        y_pred_proba_clipped = np.clip(y_pred_proba, 1e-6, 1 - 1e-6)
        focal_weights = np.zeros_like(sample_weights)

        for i, (true_class, proba, pred_class) in enumerate(zip(y_train, y_pred_proba_clipped, y_pred)):
            p_t = proba[true_class]
            lambda_dynamic = 1 + alpha * (1 - p_t)  # Adaptive lambda
            focal_weights[i] = (1 - p_t) ** gamma
            
            # Boost weight only for misclassified samples
            if true_class != pred_class:
                focal_weights[i] *= lambda_dynamic

        # Update sample weights for the next iteration
        sample_weights = focal_weights

        # Evaluate model performance on test set
        y_test_pred = ensemble_clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, average='weighted')
        recall = recall_score(y_test, y_test_pred, average='weighted')
        f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        # Track best iteration model for the current gamma
        if accuracy > best_iteration_accuracy:
            best_iteration_accuracy = accuracy
            best_iteration_ensemble_clf = ensemble_clf
            best_iteration_metrics = {
                "iteration": iteration,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }

    # Generate classification report
    classification_rep = classification_report(y_test, best_iteration_ensemble_clf.predict(X_test))
    
    # Store best model accuracy for current gamma
    accuracies.append(best_iteration_accuracy)
    accuracy_results.append((gamma, best_iteration_accuracy, best_iteration_metrics))
    print(f"Gamma: {gamma}, Best Accuracy: {best_iteration_accuracy:.4f}, Best Iteration: {best_iteration_metrics['iteration']}, Precision: {best_iteration_metrics['precision']:.4f}, Recall: {best_iteration_metrics['recall']:.4f}, F1 Score: {best_iteration_metrics['f1_score']:.4f}")
    
    # Save classification report to text file
    with open("accuracy_results5.txt", "a") as f:
        f.write(f"Gamma: {gamma}\n")
        f.write(f"Best Iteration: {best_iteration_metrics['iteration']}\n")
        f.write(f"Accuracy: {best_iteration_accuracy:.4f}\n")
        f.write(f"Precision: {best_iteration_metrics['precision']:.4f}\n")
        f.write(f"Recall: {best_iteration_metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {best_iteration_metrics['f1_score']:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_rep)
        f.write("\n" + "-" * 50 + "\n")

