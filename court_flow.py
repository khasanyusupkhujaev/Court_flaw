import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_database(file_path):
    try:
        df = pd.read_excel('test_db.xlsx')
        print("Database loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading database: {e}")
        return None

def preprocess_data(df):
    for col in ['Plaintiff\'s claim', 'Approved amount']:
        df[col] = df[col].replace(r'[^\d.]', '', regex=True)  
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)  
    
    df['Success Rate'] = df.apply(
        lambda row: row['Approved amount'] / row['Plaintiff\'s claim'] if row['Plaintiff\'s claim'] > 0 else 0, axis=1
    )
    df['Success Rate'] = df['Success Rate'].clip(upper=1.0)  
    
    df['Outcome'] = df['Success Rate'].apply(
        lambda x: 'Full' if x >= 0.95 else 'Partial' if x > 0 else 'Rejected'
    )
    
    return df


def predict_outcome(df, user_problem, user_claim_amount, user_location):
    dispute_keywords = user_problem.lower().split()
    df['Similarity'] = df['Nature of dispute'].apply(
        lambda x: sum(1 for word in dispute_keywords if word in x.lower()) / len(dispute_keywords)
    )

    print("\n=== Similarity Scores ===")
    print(df[['Nature of dispute', 'Similarity']].head()) 

    similar_cases = df[
        (df['Similarity'] > 0.5) &  
        (df['Location'].str.lower() == user_location.lower())
    ]
    
    if similar_cases.empty:
        return {
            "Message": "No similar cases found in the database.",
            "Success Rate": 0,
            "Predicted Approved Amount": 0,
            "Legal Reasoning": "Insufficient data for prediction."
        }
    
    avg_success_rate = similar_cases['Success Rate'].mean()
    predicted_amount = user_claim_amount * avg_success_rate
    
    common_outcome = similar_cases['Outcome'].mode()[0]
    reasoning_samples = similar_cases['Key legal reasoning'].head(3).tolist()  
    
    return {
        "Success Rate": round(avg_success_rate * 100, 2),  
        "Predicted Approved Amount": round(predicted_amount, 2),
        "Likely Outcome": common_outcome,
        "Similar Cases Found": len(similar_cases),
        "Sample Legal Reasoning": reasoning_samples
    }

def main():
    file_path = "test_db.xlsx"  
    df = load_database(file_path)
    if df is None:
        return
    
    df = preprocess_data(df)
    print("Preprocessed data preview:")
    print(df.head())
    
    user_problem = "The case involves a contractual dispute over a cotton procurement contract where the plaintiff delivered cotton but the defendant failed to pay fully."
    user_claim_amount = 250000000  # 250 million UZS
    user_location = "Tashkent"
    
    result = predict_outcome(df, user_problem, user_claim_amount, user_location)
    
    print("\n=== Prediction Results ===")
    for key, value in result.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()