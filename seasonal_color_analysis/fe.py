import os
from io import BytesIO
from PIL import Image, ImageDraw

import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd

from seasonal_color_analysis.core.classification import ImageSeasonClassifier


# config
FACE_EMBEDDER = os.environ["FACE_EMBEDDER"]
CLASSIFIER_PATH = os.environ["CLASSIFIER_PATH"]


CLASSIFIER = ImageSeasonClassifier.load(CLASSIFIER_PATH, FACE_EMBEDDER)


@st.cache_data
def predict(img_bytes: bytes) -> tuple[np.ndarray | None, dict[str, float] | None, np.ndarray | None]:
    with Image.open(BytesIO(img_bytes)) as img:
        batch_boxes, proba_dicts, np_season_embeddings = CLASSIFIER.predict([img])
        return batch_boxes[0], proba_dicts[0], np_season_embeddings[0]


@st.cache_data
def draw_bbox(img_bytes: bytes, bbox: np.ndarray) -> Image:
    with Image.open(BytesIO(img_bytes)) as img:
        _img = img.copy()
        draw = ImageDraw.Draw(_img)
        draw.rectangle(bbox.tolist(), outline="green", width=img.size[0] // 100)
        return _img


@st.cache_data
def draw_radar(probs: dict):
    df = pd.DataFrame(dict(
        season=list(probs.keys()),
        prob=list(probs.values()))
    )
    fig = px.line_polar(df, r="prob", theta="season", line_close=True)
    fig.update_traces(fill="toself")
    return fig


# App title
st.set_page_config(
    page_title="MangoApp",
    page_icon="🥭"
)

with st.sidebar:
    st.title(":orange[Mango]App 0.1.0")
    #st.image(SIDEBAR_IMAGE)

    # Useful links
    st.markdown('[![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/lajota13/seasonal-color-analysis)')

st.title("Seasonal color analysis with :orange[Mango]App")
st.write(
        """
        Ever wondered which colors suit you the best? 
        
        Seasonal color analysis associates different color palettes to the four seasons of the year, 
        claiming everyone can be assigned to one of the seasons, and so to specific colors.
        
        Try **MangoApp** and discover your season with the help of AI!
        """
    )

img_stream = st.file_uploader(
    "Upload a selfie", 
    type=["png", "jpg"], 
    help="""
    The photo you upload should portray just your face on the foreground with natural lighting for best accuracy. 
    
    Disclaimer: in order to ensure everyone's privacy, MangoApp will never store permanently your photo on its servers.
    """
)

if img_stream is not None:
    img_bytes = img_stream.getvalue()
    bbox, proba_dict, np_season_embeddings = predict(img_bytes)
    if bbox is None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_bytes, caption="Your image")
        with col2:
            st.write("⚠️It was not possibile to detect any face in your image, try uploading another one ⚠️")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_bytes, caption="Your image")
        with col2:
            img_w_bbox = draw_bbox(img_bytes, bbox)
            st.image(np.array(img_w_bbox), caption="Detected face")
        st.header("Your result")
        col3, col4 = st.columns(2)
        with col3:
            st.text(proba_dict)
            fig = draw_radar(proba_dict)
            st.plotly_chart(fig, use_container_width=True, theme=None)

