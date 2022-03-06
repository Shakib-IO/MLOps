# import Libraries 

import io
from urllib import request
import requests
from PIL import Image
from requests_toolbelt import MultipartEncoder #https://pypi.org/project/requests-toolbelt/
import streamlit as st

# Now lets set the API endpoint 
backend = "http://fastapi:8000/segmentation"

def process(image, server_url:str):
    m = MultipartEncoder(fields= {"files": ("filename", image, "image/jpeg")})

    r = requests.post(server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000)
    return r

# Build the UI Layout
st.title("Image Segmentation using DeepLabV3")

st.write(
        """Obtain semantic segmentation maps of the image in input via DeepLabV3 implemented in PyTorch.
         This streamlit example uses a FastAPI service as backend.
         Visit this URL at `:8000/docs` for FastAPI documentation."""
)

input_image = st.file_uploader("Insert an Image") # Upload image
st.progress(0)
if st.button("Get segmentation map"):
    """ Make two columns separate one column will
    display the orginal image and another one will display
    Segmented image. """
    col1, col2 = st.columns(2)

    if input_image:
        segments = process(input_image, backend)
        original_image = Image.open(input_image).convert("RGB")
        segmented_image = Image.open(io.BytesIO(segments.content)).convert("RGB")
        col1.header("Original")
        col1.image(original_image, use_column_width=True)
        col2.header("Segmented")
        col2.image(segmented_image, use_column_width=True)

    else:
        st.write("Please Insert an Image")