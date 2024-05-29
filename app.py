import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

# Load clustering results
clustering_results = pd.read_csv('/mnt/data/clustering_results.csv')

# Plot clustering results using numerical features
st.title('Clustering Results Visualization')
st.write('This is a visualization of the clustering results.')
fig, ax = plt.subplots()
scatter = ax.scatter(clustering_results.index, clustering_results['cluster'], c=clustering_results['cluster'], cmap='viridis')
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
st.pyplot(fig)

# Interface for prediction
st.title('Predict Football Player Categories')
st.write('Enter the features of the player to predict their category.')

feature_values = {}
feature_names = ['height', 'age', 'appearance', 'goals', 'assists', 'current_value', 'position_encoded']

for feature in feature_names:
    feature_values[feature] = st.number_input(feature)

if st.button('Predict'):
    payload = {feature: feature_values[feature] for feature in feature_names}
    st.write(f"Sending payload: {payload}")  # Debugging: Show the payload
    response = requests.post('http://127.0.0.1:8000/predict', json=payload)
    
    if response.status_code == 200:
        prediction = response.json().get('prediction')
        st.write(f'The predicted category is: {prediction}')
    else:
        st.write(f"Error: {response.status_code}")
        st.write(response.text)
