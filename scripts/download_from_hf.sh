#!/bin/bash

# Simple script to download repo data form Hugging Face

# download season classifier
wget https://huggingface.co/datasets/lajota13/lfw_facenet_embeddings/resolve/main/classifier_weights_v1.pt -O data/classifier_weights_v1.pt

# download face color embeddings
wget https://huggingface.co/datasets/lajota13/lfw_facenet_embeddings/resolve/main/lfw-colors.parquet -O

# download facenet embeddings
wget https://huggingface.co/datasets/lajota13/lfw_facenet_embeddings/resolve/main/lfw_facenet_embeddings.parquet -O data/lfw_facenet_embeddings.parquet

# download seasons labels obtained with Label Spreading algorithm
wget https://huggingface.co/datasets/lajota13/lfw_facenet_embeddings/resolve/main/lfw_facenet_embeddings_label_spreading.parquet -O data/lfw_facenet_embeddings_label_spreading.parquet

# download train season embeddings
wget https://huggingface.co/datasets/lajota13/lfw_facenet_embeddings/resolve/main/lfw_season_embeddings_train.parquet -O data/lfw_season_embeddings_train.parquet

# download test season embeddings
wget https://huggingface.co/datasets/lajota13/lfw_facenet_embeddings/resolve/main/lfw_season_embeddings_test.parquet -O data/lfw_season_embeddings_test.parquet