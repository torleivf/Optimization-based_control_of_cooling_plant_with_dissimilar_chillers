import streamlit as st
from src.oks.OSS import plotters

st.title(
    "This is a demo of the operator support system developed for Øya cooling plant"
)

legend_fig = plotters.get_standard_plot_legends_figure()
st.pyplot(legend_fig)
