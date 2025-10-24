import streamlit as st
import pandas as pd
from utils import results_display, baser
from UIs import *

st.set_page_config(page_title="FSC HG Vovinam", layout="wide")

logo_fpt_base64 = baser.image_path_to_base64("public/fpt.svg")
logo_vovinam_base64 = baser.image_path_to_base64("public/vovinam.png") 

st.sidebar.image('public/fpt_education.png')
selected_page = st.sidebar.selectbox(
    'Chọn trang',
    ['Giáo dục thể chất','Vovinam'],
    index=0,
)
st.sidebar.title("Video mẫu")

from UIs import sidebar
sidebar.init(selected_page)

if selected_page == 'Vovinam':
    st.markdown(
        f"""
        <style>
        .fpt-logo {{
            position: fixed;
            top: 50%;
            left: 64%;
            transform: translate(-50%, -50%);
            z-index: 0;
            opacity: 0.05;
            pointer-events: none;
        }}
        .vovinam-logo {{
            position: fixed;
            top: 50%;
            left: 36%;
            transform: translate(-50%, -50%);
            z-index: 0;
            opacity: 0.05;
            pointer-events: none;
        }}
        </style>
        <div class="fpt-logo">
            <img src="data:image/svg+xml;base64,{logo_fpt_base64}" width="400">
        </div>
        <div class="vovinam-logo">
            <img src="data:image/png;base64,{logo_vovinam_base64}" width="400">
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f"""
        <style>
        .fpt-logo {{
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            z-index: 0;
            opacity: 0.05;
            pointer-events: none;
        }}

        </style>
        <div class="fpt-logo">
            <img src="data:image/svg+xml;base64,{logo_fpt_base64}" width="400">
        </div>
        """,
        unsafe_allow_html=True
    )


middle, right = st.columns([2, 1])

with middle:
    dashboard.init(selected_page)
with right:
    results_display.init()