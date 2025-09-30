from utils import temper, results_display
from core import pose_detector
import streamlit as st
import definations
import os

studentVideos, sampleVideo, studentCode = None, None, None

def onSubmit(sampleVideo, studentVideos, studentCode):
    results_display.reset()
    studentVideoUrls = []
    sampleVideoUrl = temper.getVideoTempUrl(sampleVideo)
    
    for video in studentVideos:
        studentVideoUrls.append(temper.getVideoTempUrl(video))
    
    with st.spinner("Đang xử lý...", show_time=True):
        ress = pose_detector.analyze(sampleVideoUrl, studentVideoUrls, studentCode, skip_frames=0)
    for index, res in enumerate(ress):
        results_display.add(res, index)
        
    os.remove(sampleVideoUrl)
    for url in studentVideoUrls:
        os.remove(url)
    
def init():
    global studentVideos, sampleVideo, studentCode
    
    st.sidebar.title("Video mẫu")
    sampleVideo = st.sidebar.file_uploader(
        "Chọn video mẫu (.mp4, .mov, .avi)",
        accept_multiple_files=False
    )

    if sampleVideo is not None:
        st.sidebar.success(f"Đã tải lên video mẫu: {sampleVideo.name}")
    
    st.sidebar.title("Video học sinh")
    studentCode = st.sidebar.text_input("Mã học sinh", value="fhg00000")
    studentVideos = st.sidebar.file_uploader(
        "Chọn video học sinh (.mp4, .mov, .avi)",
        accept_multiple_files=True
    )

    if studentVideos:
        st.sidebar.success(f"Đã tải lên các video học sinh: {', '.join([v.name for v in studentVideos])}")
    
    st.sidebar.button("Bắt đầu chấm điểm", on_click=lambda:onSubmit(
        sampleVideo, studentVideos, studentCode
    ))

    