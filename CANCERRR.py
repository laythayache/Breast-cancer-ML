import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Step 1: Load Data
# Load the CSV file
df = pd.read_csv('Cancer_Data.csv')

# Step 2: Data Preprocessing
# Drop irrelevant columns
df_cleaned = df.drop(['id', 'Unnamed: 32'], axis=1)

# Encode diagnosis (M = 1, B = 0)
df_cleaned['diagnosis'] = LabelEncoder().fit_transform(df_cleaned['diagnosis'])

# Split features and target
X = df_cleaned.drop('diagnosis', axis=1)
y = df_cleaned['diagnosis']

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 4: Build and Train Model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

# Step 6: Display Results
print(f'Accuracy: {accuracy:.4f}')
print(f'ROC-AUC Score: {roc_auc:.4f}')
print('\nClassification Report:\n', report)
print('\nConfusion Matrix:\n', confusion)

# Step 7: Save Model (Optional)
import joblib
joblib.dump(model, 'cancer_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
