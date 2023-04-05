#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# Created By   : Charley ∆. Lebarbier
# Date Created : Tuesday 28 Mar. 2023
# ==============================================================================


import numpy as np
import random
import streamlit as st

from io import BytesIO
from PIL import Image
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------- #

## -- Download the segmented image
def download_seg_image(img):
    """Transform a numpy Array into a png"""

    buf = BytesIO()
    saving_img = Image.fromarray(img)       # Convert nparray in Image
    saving_img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


@st.cache(suppress_st_warning=True)
def segmentation(k_cluster: int, upload):
    """Use to segment an image in different cluster

    Parameters
    ----------
    k_cluster: int, required
        Cluster number for the segmentation
    upload: required
        Image format (png, jpg, jpeg) where to apply the segmentation
    """

    ## -- get uploaded image and display it (Original)
    original_image = Image.open(upload)

    col1.write("Original Image :camera:")
    col1.image(original_image)

    ## -- Load Image and transform to a 2D numpy array for KMeans.
    image = np.array(original_image)
    w, h, d = tuple(image.shape)
    image_array = np.reshape(image, (w * h, d))

    ## -- Segment the image with K-Means
    segmentation = KMeans(n_clusters=k_cluster).fit(image_array)
    centers = segmentation.cluster_centers_
    centers = np.array(centers, dtype='uint8')

    colors = [[random.randint(0,255) for i in range(3)] for center in centers]

    img_seg = np.zeros((w * h, 3), dtype='uint8')
    for ix in range(img_seg.shape[0]):
        img_seg[ix] = colors[segmentation.labels_[ix]]

    img_seg = img_seg.reshape(w, h, -1)

    ## -- Display Segmentation
    col2.write("Segmented Image :wrench:")
    col2.image(img_seg, clamp=True, channels='RGB')

    ## -- Download Segmented Image
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download Segmented Image", 
                               download_seg_image(img_seg), 
                               "img_seg.png", "image/png")


################################################################################
################################# STREAMLIT ####################################

## -- METADATA
st.set_page_config(
                    page_title = "Traitement d'images",
                    page_icon = ":camera:",
                    layout = "wide")

## -- BACKGROUND LAYER
page_bg_img = f"""
  <style>
    .stApp {{
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    }}
  </style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

## -- SIDEBAR
st.sidebar.write("## Upload and Configuration :gear:")
k_cluster = st.sidebar.slider('Select K_Cluster for the Segmentation', 1, 50)
my_upload = st.sidebar.file_uploader("Upload an image", 
                                     type=["png", "jpg", "jpeg"])

## -- MAIN CONTENT
st.header("Segment your image with clustering k-means")
st.write("Try uploading an image to watch the segmentation. Full quality \
         images can be downloaded from the sidebar after processing.")

col1, col2 = st.columns(2)

if my_upload is not None:
    segmentation(k_cluster, upload=my_upload)
else:
    st.subheader("No image uploaded yet")
