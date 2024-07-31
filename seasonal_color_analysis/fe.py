from io import BytesIO
from PIL import Image, ImageDraw

import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from seasonal_color_analysis.core.classification import ImageSeasonClassifier


# config
FACE_EMBEDDER = "vggface2"
CLASSIFIER_PATH = "data/classifier_weights_v1.pt"
SEASON_DESCRIPTION = {
    "winter": "w",
    "summer": "s",
    "spring": "s",
    "autumn": "a"
}

CLASSIFIER = ImageSeasonClassifier.load(CLASSIFIER_PATH, FACE_EMBEDDER)


@st.cache_data
def predict(img_bytes: bytes) -> tuple[np.ndarray | None, dict[str, float] | None, np.ndarray | None]:
    with Image.open(BytesIO(img_bytes)) as img:
        batch_boxes, proba_dicts, np_season_embeddings = CLASSIFIER.predict([img.convert("RGB")])
        return batch_boxes[0], proba_dicts[0], np_season_embeddings[0]


@st.cache_data
def draw_bbox(img_bytes: bytes, bbox: np.ndarray) -> Image:
    with Image.open(BytesIO(img_bytes)) as img:
        _img = img.copy()
        draw = ImageDraw.Draw(_img)
        draw.rectangle(bbox.tolist(), outline="green", width=img.size[0] // 100)
        return _img


@st.cache_data
def draw_barplot(probs: dict):
    fig = px.bar(
        pd.DataFrame(
            {
                "season": list(probs.keys()), 
                "probability": [100 * p for p in probs.values()]
            }
        ), 
        x='season', 
        y='probability'
    )
    return fig


@st.cache_data
def draw_embedding(np_season_embedding: np.ndarray):
    pass


# App title
st.set_page_config(
    page_title="MangoApp",
    page_icon="🥭"
)

with st.sidebar:
    st.title(":orange[Mango]App 0.1.0")
    #st.image(SIDEBAR_IMAGE)
    st.caption(
        """
        Ever wondered which colors suit you the best? 
        
        Seasonal color analysis associates different color palettes to the four seasons of the year, 
        claiming everyone can be assigned to one of the seasons, and so to specific colors.
        
        Try **MangoApp** and discover your season with the help of AI!
        """
    )

    # Useful links
    st.markdown('[![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/lajota13/seasonal-color-analysis)')

st.title("Seasonal color analysis with :orange[Mango]App")

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
    bbox, proba_dict, np_season_embedding = predict(img_bytes)
    if bbox is None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_bytes, caption="Your image")
        with col2:
            st.write("⚠️\n\nIt was not possibile to detect any face in your image, try uploading another one\n\n⚠️")
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
            with st.container(border=True):
                fig1 = draw_barplot(proba_dict)
                st.plotly_chart(fig1, use_container_width=True)
        #with col4:
        #    with st.container(border=True):
        #        fig2 = draw_embedding(np_season_embedding)
        #        st.plotly_chart(fig2, use_container_width=True)
        seasons = list(proba_dict.keys())
        probs = list(proba_dict.values())
        # describe the most likely season
        most_likely_season = seasons[np.argsort(probs)[-1]]
        most_likely_prob = np.sort(probs)[-1]
        st.header(most_likely_season)
        st.write(f"You are most likely a **{most_likely_season}**, with a probability of {int(100 * most_likely_prob)} %!")
        st.write(SEASON_DESCRIPTION[most_likely_season])
        # describe the second most likely season
        second_most_likely_season = seasons[np.argsort(probs)[-2]]
        second_most_likely_prob = np.sort(probs)[-2]
        if most_likely_prob < 0.9:
            st.header(second_most_likely_season)
            st.write(f"However, you could be a **{second_most_likely_season}** as well (probability of {int(100 * second_most_likely_prob)} %)")
            st.write(SEASON_DESCRIPTION[second_most_likely_season])
