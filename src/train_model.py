import os
import joblib
from sklearn.linear_model import LogisticRegression

def train(X_train, y_train, model_path='models/model.pkl'):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    # Create 'models/' folder if it doesn't exist
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save model
    joblib.dump(model, model_path)
    return model
