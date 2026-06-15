import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px

st.set_page_config(page_title="Spotify Data Pipeline Insights", layout="wide")

# Load data from the ETL output
@st.cache_data
def get_data():
    conn = sqlite3.connect("spotify_data.db")
    df = pd.read_sql("SELECT * FROM tracks", conn)
    conn.close()
    return df

try:
    df = get_data()
    
    st.title("🎵 Spotify Tracks Analytics Dashboard")
    st.markdown("Metrics and distributions derived from the automated ETL pipeline.")
    
    # --- KPI METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Unique Tracks", f"{len(df):,}")
    col2.metric("Total Genres", df['track_genre'].nunique())
    col3.metric("Avg Popularity", f"{df['popularity'].mean():.1f}/100")
    col4.metric("Avg Danceability", f"{df['danceability'].mean():.2f}")
    
    st.divider()
    
    # --- CHARTS SECTION ---
    left_col, right_col = st.columns(2)
    
    with left_col:
        st.subheader("Audio Feature Correlation")
        # Scatter plot for Danceability vs Energy
        selected_genre = st.selectbox("Filter by Genre", ["All"] + list(df['track_genre'].unique()[:20]))
        plot_df = df if selected_genre == "All" else df[df['track_genre'] == selected_genre]
        
        fig = px.scatter(plot_df, x="danceability", y="energy", color="valence", 
                         hover_name="track_name", title=f"Danceability vs Energy ({selected_genre})")
        st.plotly_chart(fig, use_container_width=True)
        
    with right_col:
        st.subheader("Top Genres by Average Popularity")
        genre_pop = df.groupby('track_genre')['popularity'].mean().sort_values(ascending=False).head(10).reset_index()
        fig2 = px.bar(genre_pop, x='popularity', y='track_genre', orientation='h', color='popularity',
                      labels={'popularity': 'Avg Popularity', 'track_genre': 'Genre'})
        st.plotly_chart(fig2, use_container_width=True)
        
    # --- SQL QUERY / ML DOWNSTREAM VIEW ---
    st.divider()
    st.subheader("🛠️ Downstream Model Data Exporter")
    st.write("Filter data using SQL queries for your ML models.")
    
    min_popularity = st.slider("Minimum Popularity for Training Set", 0, 100, 40)
    
    # Simple query interface
    conn = sqlite3.connect("spotify_data.db")
    query = f"SELECT track_name, artists, danceability, energy, loudness, valence, popularity FROM tracks WHERE popularity >= {min_popularity} ORDER BY popularity DESC"
    filtered_df = pd.read_sql(query, conn)
    conn.close()
    
    st.dataframe(filtered_df.head(100), use_container_width=True)
    st.download_button("Download Cleaned ML Feature Set", filtered_df.to_csv(index=False), "ml_features.csv", "text/csv")

except FileNotFoundError:
    st.error("⚠️ Cleaned data database not found. Please run the `etl_pipeline.py` script first to generate the data.")