import os
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
CLASSIFIER_PATH = os.path.join("data", "classifier_weights_v1.pt")
SEASON_EMBEDDINGS_PATH = os.path.join("data", "lfw_season_embeddings_train.parquet")
SEASON_DESCRIPTION_PATH = os.path.join("data", "seasons_descriptions")
FOREGROUND_IMAGE_PATH = os.path.join("data", "embedding_projector_label_spreading/embeddings.png")

CLASSIFIER = ImageSeasonClassifier.load(CLASSIFIER_PATH, FACE_EMBEDDER)


#@st.cache_data
def get_season_description(season: str) -> tuple[str, str]:
    p = os.path.join(SEASON_DESCRIPTION_PATH, season + ".md")
    with open(p) as fid:
        s = fid.read()
    summary, detail = s.split("\n\n", 1)
    return summary, detail


@st.cache_data
def predict(img_bytes: bytes) -> tuple[np.ndarray | None, dict[str, float], np.ndarray, np.ndarray]:
    with Image.open(BytesIO(img_bytes)) as img:
        batch_boxes, proba_dicts, np_season_embeddings, np_facenet_embeddings = CLASSIFIER.predict([img.convert("RGB")])
        return batch_boxes[0], proba_dicts[0], np_season_embeddings[0], np_facenet_embeddings[0]


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
    df = pd.read_parquet(SEASON_EMBEDDINGS_PATH)[
        ["name", "macroseason", "x", "y"]
    ].rename(columns={"macroseason": "season"})
    df["size"] = 1e-2
    df.loc[len(df), :] = ["You", most_likely_season, np_season_embedding[0], np_season_embedding[1], 1e-1]
    fig = px.scatter(
        df, 
        x="x", 
        y="y", 
        color="season", 
        size="size",
        hover_name='name',
        hover_data={"size": False},
        color_discrete_sequence=px.colors.qualitative.Light24
    )
    fig.add_annotation(
        x=np_season_embedding[0],
        y=np_season_embedding[1],
        text="You",
        showarrow=True,
        xanchor="right",
        font=dict(size=50)
    )
    fig.update_layout(xaxis_visible=False, yaxis_visible=False)
    return fig


# App title
st.set_page_config(
    page_title="MangoApp",
    page_icon="🥭"
)

with st.sidebar:
    st.title(":orange[Mango]App 0.1.0β")
    #st.image(SIDEBAR_IMAGE)
    st.caption(
        """
        Ever wondered which colors suit you the best? 
        
        In its simplest form, seasonal color analysis associates different color palettes to the four seasons of the year, 
        claiming everyone can be assigned to one of them, and so to specific colors.

        A detailed and professional analysis generally requires a skilled specialist, 
        but if you are down to have just some fun and get some hints on your season, 
        try **MangoApp**: the AI-powered tool for seasonal color analysis!
        """
    )

    # Useful links
    st.markdown('[![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/lajota13/seasonal-color-analysis)')

st.title("Seasonal color analysis with :orange[Mango]App")
st.image(FOREGROUND_IMAGE_PATH)

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
    bbox, proba_dict, np_season_embedding, np_facenet_embedding = predict(img_bytes)
    if bbox is None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_bytes, caption="Your image")
        with col2:
            st.write("⚠️\n\nIt was not possibile to detect any face in your image, try uploading another one\n\n⚠️")
    else:
        seasons = list(proba_dict.keys())
        probs = list(proba_dict.values())
        most_likely_season = seasons[np.argsort(probs)[-1]]
        most_likely_prob = np.sort(probs)[-1]
        second_most_likely_season = seasons[np.argsort(probs)[-2]]
        second_most_likely_prob = np.sort(probs)[-2]
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_bytes, caption="Your image")
        with col2:
            img_w_bbox = draw_bbox(img_bytes, bbox)
            st.image(np.array(img_w_bbox), caption="Detected face")
        
        st.header("Your result")

        st.caption("Season probability")
        fig1 = draw_barplot(proba_dict)
        st.plotly_chart(fig1, use_container_width=True)
        st.caption("Where you are in the seasonal-color space")
        fig2 = draw_embedding(np_season_embedding)
        st.plotly_chart(fig2, use_container_width=True)
        
        # describe the most likely season
        most_likely_summary, most_likely_detail = get_season_description(most_likely_season)
        with st.expander(
            f"You are most likely a **{most_likely_season.capitalize()}**," 
            f"with a probability of {int(100 * most_likely_prob)} %!"
            f"\n\n{most_likely_summary}"
        ):
            st.markdown(most_likely_detail)
        
        # describe the second most likely season
        if most_likely_prob < 0.9:
            second_most_likely_summary, second_most_likely_detail = get_season_description(second_most_likely_season)
            with st.expander(
                f"However you could be a **{most_likely_season.capitalize()}** too," 
                f"with a probability of {int(100 * most_likely_prob)} %."
                f"\n\n{second_most_likely_summary}"
            ):
                st.markdown(second_most_likely_detail)
