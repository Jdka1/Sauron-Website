import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import requests
from io import BytesIO
import zipfile
import io
import streamlit as st
from data_analysis import get_targetability_df, merge_latlong
import folium
import numpy as np



with st.sidebar:
        st.header("Variables")
        crime_weight = st.slider("City Crime Rate Importance For Targetability Weight", min_value=1.0, max_value=5.0, step=0.1, value=1.0)
        wealth_weight = st.slider("City Wealth Importance For Targetability Weight", min_value=1.0, max_value=5.0, step=0.1, value=1.0)
        minimum_population = st.slider("Minimum Population", min_value=0, max_value=1000000, step=1000, value=50000)


def map_to_rgb(viability):
    # Linear mapping from viability to RGB values
    r = 0
    g = int(255 * (1 - (viability/100)))
    b = 0  # Set blue component to 0 for simplicity
    alpha = (viability/150)
    return (r, g, b, alpha)

st.title('Most Targetable Cities')

df = get_targetability_df(crime_weight=crime_weight, wealth_weight=wealth_weight, minimum_population=minimum_population)
st.dataframe(df)

map_df = merge_latlong(df)
map_df['RGB'] = map_df['Target viability'].apply(map_to_rgb)
map_df['size'] = map_df['Target viability'] * 200
st.map(map_df, color='RGB', size='size', latitude='lat', longitude='lng', zoom=3)