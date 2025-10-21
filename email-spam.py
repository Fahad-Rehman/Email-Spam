import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc
)

data = pd.read_csv('emails.csv')

# Initial data exploration
print("Columns:", list(data.columns[:10]))
print("Shape:", data.shape)

# Drop first column (email index or ID)
data = data.drop(data.columns[0], axis=1)

# Separate features and target
X = data.drop('Prediction', axis=1)
y = data['Prediction']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Model training
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Cross-validation score
lr_cv = cross_val_score(lr, X, y, cv=5, scoring='accuracy').mean()

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Cross-validation score
rf_cv = cross_val_score(rf, X, y, cv=5, scoring='accuracy').mean()

#Evaluation
def evaluate_model(name, y_true, y_pred):
    print(f"\n --- {name} Evaluation ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

evaluate_model("Logistic Regression", y_test, lr_pred)
evaluate_model("Random Forest", y_test, rf_pred)

print("\nCross-Validation Accuracy:")
print(f"Logistic Regression: {lr_cv:.4f}")
print(f"Random Forest: {rf_cv:.4f}")

#Check for overfitting
train_acc_rf = accuracy_score(y_train, rf.predict(X_train))
test_acc_rf = accuracy_score(y_test, rf_pred)
print(f"\nRandom Forest Training Accuracy: {train_acc_rf:.4f}")
print(f"Random Forest Testing Accuracy: {test_acc_rf:.4f}")

#Confusion Matrix
cm = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#ROC Curve and AUC
# Predict probabilities
lr_probs = lr.predict_proba(X_test)[:, 1]
rf_probs = rf.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)

auc_lr = auc(fpr_lr, tpr_lr)
auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(6, 5))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.3f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

#Top Features from Logistic Regression
coeffs = lr.coef_[0]
top_spam = np.argsort(coeffs)[-15:]
top_ham = np.argsort(coeffs)[:15]

print("\nTop Spam Indicator Words:")
print([X.columns[i] for i in top_spam])

print("\nTop Non-Spam (Ham) Indicator Words:")
print([X.columns[i] for i in top_ham])

# Visualize them
plt.figure(figsize=(8, 5))
plt.barh([X.columns[i] for i in top_spam], coeffs[top_spam], color='red')
plt.title("Top Spam Indicator Words (Logistic Regression)")
plt.xlabel("Coefficient Value")
plt.show()

