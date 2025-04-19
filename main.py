import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    df = movies.merge(credits, on='title')
    df = df[['title', 'budget', 'revenue', 'genres', 'runtime', 'release_date', 'cast', 'crew']]
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_month'] = df['release_date'].dt.month
    df['success'] = (df['revenue'] > 2 * df['budget']).astype(int)
    df.dropna(subset=['budget', 'revenue', 'runtime', 'release_month'], inplace=True)

    # Genre extraction
    df['main_genre'] = df['genres'].apply(lambda x: json.loads(x.replace("'", '"'))[0]['name'] if x != '[]' else 'Unknown')
    genre_dummies = pd.get_dummies(df['main_genre'], prefix='genre')
    df = pd.concat([df, genre_dummies], axis=1)

    # Director extraction
    def get_director(crew_str):
        try:
            crew = json.loads(crew_str.replace("'", '"'))
            for person in crew:
                if person['job'] == 'Director':
                    return person['name']
        except:
            return 'Unknown'

    df['director'] = df['crew'].apply(get_director)
    top_directors = df['director'].value_counts().head(10).index
    df['top_director'] = df['director'].apply(lambda x: x if x in top_directors else 'Other')
    director_dummies = pd.get_dummies(df['top_director'], prefix='director')
    df = pd.concat([df, director_dummies], axis=1)

    # Final features
    feature_cols = ['budget', 'runtime', 'release_month'] + list(genre_dummies.columns) + list(director_dummies.columns)
    X = df[feature_cols]
    y = df['success']

    return X, y, feature_cols, genre_dummies.columns.tolist(), director_dummies.columns.tolist()

X, y, feature_cols, genres, directors = load_data()

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# ---------------------------
# Streamlit App UI
# ---------------------------
st.title("🎬 TMDB Movie Success Predictor")

st.markdown("Fill the details below to check if your movie will be a 🎯 **Hit** or 💣 **Flop**")

# User Inputs
budget = st.number_input("Budget (USD)", min_value=10000, step=1000000, value=10000000)
runtime = st.slider("Runtime (minutes)", min_value=60, max_value=240, value=120)
release_month = st.selectbox("Release Month", list(range(1, 13)))
genre = st.selectbox("Main Genre", genres)
director = st.selectbox("Director", directors)

# Create input data
input_data = pd.DataFrame(np.zeros((1, len(feature_cols))), columns=feature_cols)
input_data['budget'] = budget
input_data['runtime'] = runtime
input_data['release_month'] = release_month
input_data[genre] = 1
input_data[director] = 1

# Predict button
if st.button("Predict Movie Success"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("🎉 Your movie is likely to be a **HIT**!")
    else:
        st.error("😢 Sorry, this movie might be a **FLOP**.")
