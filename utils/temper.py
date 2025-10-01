import os
import uuid
import shutil

def getVideoTempUrl(uploaded_file):
    temp_dir = "tmp"
    os.makedirs(temp_dir, exist_ok=True)
    
    _, ext = os.path.splitext(uploaded_file.name)
    random_name = f"{uuid.uuid4().hex}{ext}"
    
    temp_path = os.path.join(temp_dir, random_name)
    abs_path = os.path.abspath(temp_path)
    
    with open(abs_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return abs_path

def clear_tmp():
    tmp_dir = "tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

def getSampleVideoUrl(type):
    dir = "sample"
    abs_path = os.path.abspath(os.path.join(dir, type+".mp4"))
    return abs_path