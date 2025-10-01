from utils import temper, results_display
from UIs import dashboard
from core import pose_detector
import streamlit as st
import definations
import os

studentVideos, sampleVideo, studentCode, comparisionType = None,None,None,None

def onSubmit(sampleVideo, studentVideos, studentCode, type):
    results_display.reset()
    dashboard.reset()
    
    studentVideoUrls = []
    
    if not sampleVideo:
        sampleVideoUrl = temper.getSampleVideoUrl(type)
    else:
        sampleVideoUrl = temper.getVideoTempUrl(sampleVideo)
    
    for video in studentVideos:
        studentVideoUrls.append(temper.getVideoTempUrl(video))
    
    dashboard.add(sampleVideoUrl, "sample")
    for url in studentVideoUrls:
        dashboard.add(url, "student")
    
    with st.spinner("Đang xử lý...", show_time=True):
        ress = pose_detector.analyze(sampleVideoUrl, studentVideoUrls, studentCode, skip_frames=0)
    for index, res in enumerate(ress):
        results_display.add(res, index)
        
    #if (sampleVideo):
    #    os.remove(sampleVideoUrl)
    #for url in studentVideoUrls:
    #    os.remove(url)
    
def init():
    global studentVideos, sampleVideo, studentCode, comparisionType
    choosenType = None
    
    st.sidebar.image('public/fpt_education.png')
    st.sidebar.title("Video mẫu")
    comparisionType = st.sidebar.selectbox("Phương thức so sánh", [
        ["Chọn mẫu", "chooseType"],
        ["Tải lên mẫu", "selfUpload"]
    ], format_func=lambda a : a[0])
    
    if comparisionType[1] == "selfUpload":
        sampleVideo = st.sidebar.file_uploader(
            "Chọn video mẫu (.mp4, .mov, .avi)",
            type=[".mp4",".mov",".avi"],
            accept_multiple_files=False
        )
        if sampleVideo is not None:
            st.sidebar.success(f"Đã tải lên video mẫu: {sampleVideo.name}")
    else:
        choosenType = st.sidebar.selectbox("Chọn mẫu", definations.actions, format_func=lambda a:a[0])
        st.sidebar.success(f"Đã chọn mẫu: {choosenType[0]}")
        st.sidebar.warning(f"Vui lòng chọn góc quay {choosenType[2]}")
    
    st.sidebar.title("Video học sinh")
    studentCode = st.sidebar.text_input("Mã học sinh", value="fhg00000")
    studentVideos = st.sidebar.file_uploader(
        "Chọn video học sinh (.mp4, .mov, .avi)",
        type=[".mp4",".mov",".avi"],
        accept_multiple_files=True,
    )

    if studentVideos:
        st.sidebar.success(f"Đã tải lên các video học sinh: {', '.join([v.name for v in studentVideos])}")
    
    st.sidebar.button(
    "Bắt đầu chấm điểm",
    on_click=lambda: onSubmit(sampleVideo, studentVideos, studentCode, None if not choosenType else choosenType[1]),
    disabled=(not studentVideos or (choosenType is None and sampleVideo is None))
)

    