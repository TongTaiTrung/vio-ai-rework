import streamlit as st
import pandas as pd
import random
import pandas as pd
from io import BytesIO

ress = []
def add(res, index):
    ress.append(res);

def reset():
    global ress
    ress = []

def init():
    if len(ress) <= 0:
        return
    for index, res in enumerate(ress):
        df = pd.read_excel(BytesIO(res['excel_bytes']))
        styled_df = df.style.set_properties(**{
            'white-space': 'pre-wrap',
            'word-wrap': 'break-word'
        })
        st.dataframe(styled_df)
        st.image(res['chart_image'], f"Biểu đồ {index+1}")
        st.download_button(
            label=f"Tải bản báo cáo {index+1}",
            data=res['excel_bytes'],
            file_name=f"result_{index+1}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        st.markdown("---")