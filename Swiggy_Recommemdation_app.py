import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder

# Set Page Config
st.set_page_config(page_title="Swiggy Recommender", layout="wide")

# ==========================================        
# 1. DATA PROCESSING & MODEL TRAINING
# ==========================================
@st.cache_resource
def train_model():
    # Load and initial cleaning 
    df = pd.read_csv('swiggy.csv').drop_duplicates().reset_index(drop=True)

    # Handle Non-Numeric Values and Impute 
    for col in ['rating', 'cost', 'rating_count']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0)

    # Filling Categorical missing values 
    df[['city', 'cuisine']] = df[['city', 'cuisine']].fillna('Unknown')

    # Target Encoding 
    encoder = TargetEncoder(cols=['city', 'cuisine'])
    encoded_cols = encoder.fit_transform(df[['city', 'cuisine']], df['rating'])

    # Final Feature Preparation (X)
    X = pd.concat([encoded_cols, df[['rating', 'cost', 'rating_count']]], axis=1).fillna(0)

    # Scaling and Clustering 
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(scaled_data)
    
    # Save cleaned data for result mapping
    df.to_csv('cleaned_data.csv', index=False)
    
    return df, encoder, scaler, kmeans

# Initialize the engine
df, encoder, scaler, kmeans = train_model()

# ==========================================
# 2. STREAMLIT UI COMPONENTS 
# ==========================================
st.title("🍽️ Swiggy Restaurant Recommendation System")
st.markdown("Discover the best dining spots tailored to your taste and budget.")

# Sidebar for User Inputs
with st.sidebar:
    st.header("Your Preferences")
    
    # Selection filters
    user_city = st.selectbox("Select City", options=sorted(df['city'].unique()))
    user_cuisine = st.selectbox("Select Cuisine", options=sorted(df['cuisine'].unique()))
    user_budget = st.slider("Max Budget (Cost for two)", 0, 2000, 500)
    user_rating = st.slider("Minimum Rating", 0.0, 5.0, 3.5)

    predict_btn = st.button("Get Recommendations")

# ==========================================
# 3. RECOMMENDATION ENGINE
# ==========================================
    input_df = pd.DataFrame({
    'city': [user_city],
    'cuisine': [user_cuisine]
})

# Encode categorical features
encoded_input = encoder.transform(input_df)

# Add numerical features
input_features = pd.concat([
    encoded_input,
    pd.DataFrame({
        'rating': [user_rating],
        'cost': [user_budget],
        'rating_count': [df['rating_count'].median()]
    })
    ], axis=1)

    #Ensure column order matches training
input_features = input_features[['city', 'cuisine', 'rating', 'cost', 'rating_count']]

    # Scale input
scaled_input = scaler.transform(input_features)

 # 2. Predict the cluster for the user input
target_cluster = kmeans.predict(scaled_input)[0]

 # 3. Filter the dataframe for the same cluster
   # This will now work because target_cluster is a single number
recommendations = df[
    (df['cluster'] == target_cluster) &
    (df['city'] == user_city) &
    (df['cost'] <= user_budget) &
    (df['rating'] >= user_rating)
       ].sort_values(by='rating', ascending=False)

    # Handle No Results
if recommendations.empty:
        st.warning("No exact matches found in your city. Showing top-rated restaurants in your preferred category across all cities:")
        recommendations = df[
            (df['cluster'] == target_cluster) & 
            (df['rating'] >= user_rating)
        ].head(5)

    # Display Results [Source 6]
st.subheader(f"Top Recommendations for you:")
    
    # Formatting output for better readability
display_df = recommendations[['name', 'cuisine', 'rating', 'cost', 'city']].head(10)
st.table(display_df)
    
st.success(f"Found {len(recommendations)} matching restaurants in Cluster {target_cluster}!")