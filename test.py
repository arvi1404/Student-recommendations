from utils import *
import pandas as pd
import json
import joblib

with open("q2.json", "r") as f:
    current_submission = json.load(f)

label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

current_submission = pd.DataFrame([current_submission]) 
current_submission = expand_nested_column(current_submission, 'quiz', 'quiz_id')
current_submission = preprocess_data(current_submission)
current_submission = compute_features(current_submission)

# Generate insights for the user
generate_insights(current_submission,label_encoder,scaler,model)