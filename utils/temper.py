import os
import uuid

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