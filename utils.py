import pandas as pd
import re
from fuzzywuzzy import process
import numpy as np


def expand_nested_column(df, nested_col, id_col):
    """
    Expands a specified nested column in a Pandas DataFrame while keeping other columns unchanged.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        nested_col (str): The name of the column containing nested dictionaries to expand.
        id_col (str): The unique identifier column to merge back after expansion.

    Returns:
        pd.DataFrame: A DataFrame with the nested column expanded.
    """
    # Ensure the nested column is a list of dictionaries
    df[nested_col] = df[nested_col].apply(lambda x: x if isinstance(x, dict) else {})
    
    # Convert nested dictionary column to a DataFrame
    df_expanded = pd.json_normalize(df[nested_col]).add_prefix(f"{nested_col}_")
    
    # Merge expanded columns back to the original DataFrame
    df_final = df.drop(columns=[nested_col]).merge(df_expanded, left_on=id_col, right_on=f"{nested_col}_id", how='left')
    
    # Remove duplicates based on the unique identifier column
    df_final = df_final.drop_duplicates(subset=[id_col])

    return df_final


def dropNull(df): 
    # Function to drop Null columns
    null_cols = df.columns[df.isnull().all()]
    print("No. of Null Columns : ", len(null_cols))
    if len(null_cols) > 0:
        print("Dropped Null Columns : ",list(null_cols))
        df.drop(columns=null_cols, inplace=True)
    else : print("No columns were dropped")
    return df
    
def clean_text(text):
    """Lowercase, remove extra spaces, special characters, and numbers."""
    text = text.lower().strip()  # Convert to lowercase and strip spaces
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters and numbers
    return text

def generate_standardized_name(df, column_name, threshold=90):
    """
    Cleans text and applies fuzzy matching to standardize similar entries.
    
    Parameters:
        df (pd.DataFrame): The dataframe containing the column to process.
        column_name (str): The name of the column to clean and standardize.
        threshold (int): Similarity score threshold for fuzzy matching.
        
    Returns:
        pd.Series: A series with standardized values.
    """
    # Clean the column
    clean_col = f"clean_{column_name}"
    df[clean_col] = df[column_name].astype(str).apply(clean_text)

    # Get unique cleaned values
    unique_values = df[clean_col].unique()

    # Define mapping dictionary for standardization
    mapping = {}

    # Perform fuzzy matching
    for value in unique_values:
        if not mapping:  # First entry, add directly
            mapping[value] = value
            continue
        
        match_result = process.extractOne(value, mapping.keys())  # Find best match
        
        if match_result:  # Ensure we got a result
            match, score = match_result  
            mapping[value] = mapping[match] if score > threshold else value
        else:
            mapping[value] = value  # If no match, keep as is

    # Apply mapping to standardize column
    standardized_col = f"standardized_{column_name}"
    df[standardized_col] = df[clean_col].map(mapping)
    
    return df


def preprocess_data(df_original):
    # Function to clean the data 
    df = df_original.copy()

    df = dropNull(df)
    
    # Convert date columns to datetime format
    date_cols = ['submitted_at', 'created_at', 'updated_at', 'started_at', 'ended_at']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True) # COnverting everting to UTC without loss of generality
            df[col] = df[col].dt.tz_localize(None)  
    
    # Convert accuracy to numeric (0-1)
    df['accuracy'] = df['accuracy'].astype(str).str.rstrip('%').astype(float) / 100
    
    # Connvert rank_text to integer
    df['rank'] = df['rank_text'].str.extract(r'(\d+)$')
    df['rank'] = df['rank'].map(int)

    # Covert duration to numeric values (in mins)
    df[['minutes', 'seconds']] = df['duration'].str.split(':', expand=True).astype(int)
    df['duration'] = df['minutes'] + df['seconds']/60

    df = generate_standardized_name(df, "quiz_topic", threshold=90)
    df = generate_standardized_name(df, "quiz_title", threshold=90)

    numeric_cols = ['speed','final_score','negative_score','quiz_negative_marks','quiz_correct_answer_marks']
    df[numeric_cols] = df[numeric_cols].map(float)
    
    del_columns = ['rank_text','minutes', 'seconds','clean_quiz_topic','clean_quiz_title', 'quiz_title', 
                  'quiz_description', 'quiz_topic','quiz_time', 'quiz_is_published', 'quiz_created_at', 'quiz_updated_at',
                  'quiz_duration', 'quiz_end_time','quiz_shuffle', 'quiz_show_answers','quiz_lock_solutions', 'quiz_is_form', 
                  'quiz_show_mastery_option','quiz_is_custom', 'quiz_show_unanswered', 'quiz_ends_at','quiz_live_count', 
                  'quiz_coin_count','quiz_daily_date', 'quiz_max_mistake_count', 'quiz_reading_materials','source','type' ]
    print('Dropping unneccessary columns ...')
    print(del_columns)
    df.drop(columns=del_columns,inplace=True)
    
    return df

def compute_features(df):
    # 1. Total marks for quiz
    df['quiz_total'] = df['quiz_questions_count'] * df['quiz_correct_answer_marks']
    
    # 2. No. of questions attempted 
    df['questions_attempted'] = df['correct_answers'] + df['incorrect_answers']
    
    # 3. Total time taken for the quiz 
    df["total_time_taken"] = (df["ended_at"] - df["started_at"]).dt.total_seconds()
    
    # 4. Average response time for a question
    df['avg_response_time'] = df["total_time_taken"] / df['questions_attempted'].replace(0, np.nan)
    
    # 5. Normalized score (0-1)
    df["normalized_score"] = df["final_score"] / df["quiz_total"]
    
    # 6. Effective accuracy 
    df["effective_accuracy"] = (
        (df['correct_answers'] - (df['incorrect_answers'] * df['quiz_negative_marks'])
         / df['quiz_correct_answer_marks']) / df['questions_attempted']
    )
    
    # 7. Average Time Taken per Correct Answer
    df["time_per_correct"] = df["total_time_taken"] / df["correct_answers"].replace(0, np.nan)
    
    # 8. Average Time Taken per Incorrect Answer
    df["time_per_incorrect"] = df["total_time_taken"] / df["incorrect_answers"].replace(0, np.nan)
    
    # 9. Average Time Spent on Mistakes
    df["time_on_mistakes"] = (df["total_time_taken"] * df["incorrect_answers"]) / df["total_questions"]
    
    # 10. Mistake Correction Rate
    df["mistake_correction_rate"] = df["mistakes_corrected"] / df["initial_mistake_count"].replace(0, np.nan)
    
    # 11. Impact of negative marks
    df["penalty_impact"] = df["negative_score"] / (
        df["final_score"].abs() + df["negative_score"].abs()
    )
    
    # 12. Success Rate
    df["success_rate"] = df["correct_answers"] / df["questions_attempted"].replace(0, np.nan)
    
    # 13. Completion Rate
    df["completion_rate"] = df["questions_attempted"] / df["total_questions"]
    
    # 14. Quiz Difficulty Score
    def assign_difficulty(x):
        if x > 0.8:
            return 'Hard'
        elif 0.6 < x <= 0.8:
            return 'Medium'
        elif 0.3 < x <= 0.6:
            return 'Easy'
        else:
            return 'Very Easy'
    
    difficulty_mapping = {"Very Easy": 1, "Easy": 2, "Medium": 3, "Hard": 4}
    
    df["difficulty"] = df.groupby("quiz_id")["final_score"].transform(lambda x: x.mean()) / df["quiz_total"]
    df["difficulty"] = df["difficulty"].apply(assign_difficulty)
    df["difficulty"] = df["difficulty"].map(difficulty_mapping)
    
    # 15. Time Pressure Index
    df["quiz_time_pressure"] = (df["duration"] * 60) / df["quiz_questions_count"]
    
    return df

def generate_insights(user_data, label_encoder, scaler, model):
    """Generate performance insights based on model predictions, feature importance, and outlier detection."""
    features = ["accuracy", "time_per_correct", "time_per_incorrect", "success_rate",
    "difficulty", "effective_accuracy", "quiz_total", "questions_attempted",
    "total_time_taken", "avg_response_time", "completion_rate", "quiz_time_pressure"]

    dataset = pd.read_csv('clean_history_data.csv')
    user_features = user_data[features]
    user_scaled = scaler.transform(user_features)  # Apply scaling
    pred_class = model.predict(user_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_class])[0]  # Convert number back to label

    feature_importance = pd.DataFrame({"Feature": list(user_features.columns), "Importance": model.feature_importances_})
    feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

    print(f"\n**Overall Performance Assessment: {pred_label}**")

    # Identify the top 3 most important factors
    top_factors_df = feature_importance.head(3)
    top_factors = top_factors_df["Feature"].tolist()
    top_weights = top_factors_df["Importance"].values

    # Retrieve user-specific values for the top factors
    user_values = user_data[top_factors].values.flatten()
    
    # Compute median and 25th percentile for each factor
    median_values = dataset[features].median()
    threshold = dataset[features].quantile(0.50)   

    # Identify critically bad factors
    critically_bad_factors = []
    for factor in top_factors:
        user_value = user_data[factor].values[0]

        # If lower values are worse
        if factor in ["accuracy", "effective_accuracy", "success_rate", "completion_rate"]:
            if user_value < threshold[factor]:  
                critically_bad_factors.append(factor)

        # If higher values are worse
        if factor in ["time_per_correct", "time_per_incorrect"]:
            if user_value > threshold[factor]:  
                critically_bad_factors.append(factor)

    # Print key factors
    print("\n**Key Factors Affecting Your Performance:**")
    for factor in top_factors:
        print(f"- {factor.replace('_', ' ').title()} (Value: {user_data[factor].values[0]:.2f})")

    print("\n**Personalized Insights Based on Your Performance:**")

    # Generate insights based on critically bad factors
    if critically_bad_factors:
        for factor in critically_bad_factors:
            print(f"**Critical Issue: {factor.replace('_', ' ').title()}**")
            
            # Accuracy & Effective Accuracy (Group)
            if factor in ["accuracy", "effective_accuracy"]:
                print("- Your accuracy is significantly below average. Review mistakes carefully and focus on understanding weak areas.")
                print("- Consider solving easier problems first to build confidence before moving to harder ones.")
            
            # Success Rate
            if factor == "success_rate":
                print("- Your success rate is lower than most students. Try identifying which question types cause errors and focus on improving them.")

            # Completion Rate
            if factor == "completion_rate":
                print("- You are not completing quizzes consistently. Try setting a goal to finish every quiz to build stamina and improve scores.")

            # Time Per Correct Answer
            if factor == "time_per_correct":
                print("- You take longer than most students to answer correctly. Work on practicing timed quizzes to improve speed.")

            # Time Per Incorrect Answer
            if factor == "time_per_incorrect":
                print("- You spend too much time on incorrect answers. Consider skipping difficult questions and managing time better.")

            # Difficulty Level
            if factor == "difficulty":
                print("- You are struggling with harder questions. Focus on mastering foundational concepts before attempting advanced topics.")

            # Questions Attempted
            if factor == "questions_attempted":
                print("- You are attempting fewer questions than most. Increase the number of questions to improve retention and confidence.")

            # Total Time Taken
            if factor == "total_time_taken":
                print("- You are spending significantly more time per quiz than most. Try pacing yourself and practicing time management.")

            # Average Response Time
            if factor == "avg_response_time":
                print("- Your response time is higher than expected. Focus on improving recall speed and reaction time.")

            # Quiz Time Pressure
            if factor == "quiz_time_pressure":
                print("- You may be struggling under time pressure. Try relaxation techniques and practice with timed mock tests.")

    else:
        # If no critically bad factor, generate normal weighted insights
        insights = []

        if "time_per_correct" in top_factors or "accuracy" in top_factors:
            if user_data["time_per_correct"].values[0] > median_values["time_per_correct"] and user_data["accuracy"].values[0] < median_values["accuracy"]:
                insights.append("- You take longer to answer correctly, but accuracy is still low. Focus on improving both speed and precision.")
            elif user_data["time_per_correct"].values[0] < median_values["time_per_correct"] and user_data["accuracy"].values[0] > median_values["accuracy"]:
                insights.append("- Your speed is good, and accuracy is above average. Maintain this balance, but avoid rushing unnecessarily.")

        if "difficulty" in top_factors or "effective_accuracy" in top_factors:
            if user_data["difficulty"].values[0] > median_values["difficulty"] and user_data["effective_accuracy"].values[0] < median_values["effective_accuracy"]:
                insights.append("- You are attempting harder questions, but accuracy is suffering. Consider reviewing foundational topics before advancing.")

        if "questions_attempted" in top_factors or "completion_rate" in top_factors:
            if user_data["questions_attempted"].values[0] < median_values["questions_attempted"]:
                insights.append("- You are attempting fewer questions than most. Increase practice to build familiarity and confidence.")
            elif user_data["completion_rate"].values[0] < median_values["completion_rate"]:
                insights.append("- You start quizzes but do not finish them. Work on completing full quizzes for better progress tracking.")

        # Print normal insights
        if insights:
            for insight in insights:
                print(insight)
        else:
            print("- Your performance is well-balanced, but continuous practice can help maintain consistency.")

    print("\n **Next Steps:** Track your performance over time, focus on the identified weak areas, and refine your strategy accordingly!")
