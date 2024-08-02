FROM python:3.10-slim
LABEL python_version=python3.10

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*


# Poetry virtual environment setup
ENV POETRY_NO_INTERACTION=1
ENV POETRY_HOME=/opt/env
ENV POETRY_CACHE_DIR=/opt/.cache
# mandatory !!
ENV POETRY_VIRTUALENVS_IN_PROJECT=1

RUN python3 -m venv $POETRY_HOME
RUN $POETRY_HOME/bin/pip install -U pip setuptools
RUN $POETRY_HOME/bin/pip install poetry==1.8.3

ENV PATH=$PATH:$POETRY_HOME/bin

COPY pyproject.toml poetry.lock /app/ 
WORKDIR /app

# creating the poetry virtual environment
RUN poetry install --without dev --no-root

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="."

COPY . .

# download needed data from Hugging Face
RUN wget https://huggingface.co/datasets/lajota13/lfw_facenet_embeddings/resolve/main/lfw_season_embeddings_train.parquet -O data/lfw_season_embeddings_train.parquet

# download vggface2 weights for Facenet embeddings
RUN mkdir -p ~/.cache/torch/checkpoints
RUN wget https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt -O ~/.cache/torch/checkpoints/20180402-114759-vggface2.pt

HEALTHCHECK CMD curl --fail http://localhost:$PORT/_stcore/health

SHELL ["/bin/bash", "-c"]

ENTRYPOINT streamlit run seasonal_color_analysis/fe.py --server.port=$PORT --server.address=0.0.0.0