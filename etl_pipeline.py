import os
import pandas as pd
import sqlite3

def ingest_data(file_path):
    """Extract: Load the dataset."""
    print("🔄 Ingesting dataset...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file not found at {file_path}")
    return pd.read_csv(file_path)

def transform_data(df):
    """Transform: Clean and prepare features for analysis/ML."""
    print("🧹 Transforming data...")
    
    # 1. Drop duplicates and unnecessary index columns
    df = df.drop_duplicates(subset=['track_id'])
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    # 2. Handle missing values
    df = df.dropna(subset=['track_name', 'artists', 'album_name'])
    
    # 3. Feature Engineering: Convert duration from ms to minutes
    df['duration_min'] = df['duration_ms'] / 60000
    df = df.drop(columns=['duration_ms'])
    
    # 4. Standardize Text Data
    df['track_genre'] = df['track_genre'].str.lower().str.strip()
    
    # 5. Ensure correct data types
    df['explicit'] = df['explicit'].astype(int) # Convert boolean to 1/0 for ML readiness
    
    return df

def load_data(df, db_path="spotify_data.db", csv_path="cleaned_tracks.csv"):
    """Load: Save to CSV for Streamlit and SQLite for downstream queries."""
    print("💾 Loading data to destinations...")
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    
    # Save to SQLite for SQL analysis/ML feature store
    conn = sqlite3.connect(db_path)
    df.to_sql("tracks", conn, if_exists="replace", index=False)
    conn.close()
    
    print(print("✅ ETL Pipeline executed successfully!"))

if __name__ == "__main__":
    # Point this to your uploaded file path
    INPUT_FILE = "dataset.csv" 
    
    raw_data = ingest_data(INPUT_FILE)
    cleaned_data = transform_data(raw_data)
    load_data(cleaned_data)