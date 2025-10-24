from utils import temper
import os
import streamlit as st

videoUrls = []
def init(page):
    st.title("FSC HAG VOVINAM" if page=='Vovinam' else "FSC Giáo Dục Thể Chất")
    st.caption("Hệ thống tự chấm điểm các động tác trong môn võ Vovinam" if page=='Vovinam' else "Hệ thống tự chấm điểm động tác giáo dục thể chất")
    for index, url in enumerate(videoUrls):
        if (url[1] == 'sample'):
            st.markdown("#### Video mẫu")
        else:
            st.markdown(f"#### Video học sinh ({index})")
        st.video(url[0])
        st.markdown('---')

def add(url, type):
    global videoUrls
    videoUrls.append([url, type])

def reset():
    global videoUrls
    temper.clear_tmp()
    videoUrls = []