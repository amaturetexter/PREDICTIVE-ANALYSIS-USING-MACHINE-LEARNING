import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate(model, X_test, y_test, report_path='outputs/evaluation_report.txt'):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    os.makedirs(os.path.dirname(report_path), exist_ok=True)  # auto-create folder

    with open(report_path, 'w') as f:
        f.write(f"Accuracy: {acc:.2f}\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    print("Evaluation report saved.")
