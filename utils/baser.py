import base64
def image_path_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()