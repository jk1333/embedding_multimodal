import base64
import typing
import sys
import os
from google.cloud import aiplatform
from google.protobuf import struct_pb2
import requests
import streamlit as st
from PIL import Image
from dataclasses import dataclass

st.set_page_config(page_title='Multimodal Embedding API test', 
                    page_icon=None, 
                    layout="wide", 
                    initial_sidebar_state="auto", 
                    menu_items=None)

PROJECT_ID = sys.argv[1]
DEPLOYED_INDEX_ID = "deployed_index_id_unique"
INDEX_ENDPOINT = 'projects/1045259343465/locations/us-central1/indexEndpoints/116990236218621952' #'projects/{ID}/locations/us-central1/indexEndpoints/{E-ID}'
NUM_NEIGHBORS = 20

@st.cache_resource
def get_index_endpoint():
    return aiplatform.MatchingEngineIndexEndpoint(INDEX_ENDPOINT)
index_endpoint = get_index_endpoint()

class EmbeddingResponse(typing.NamedTuple):
    @dataclass
    class VideoEmbedding:
        start_offset_sec: int
        end_offset_sec: int
        embedding: typing.Sequence[float]

    text_embedding: typing.Sequence[float]
    image_embedding: typing.Sequence[float]
    #video_embeddings: typing.Sequence[VideoEmbedding]

def load_image_bytes(image_uri: str) -> bytes:
    """Load image bytes from a remote or local URI."""
    image_bytes = None
    if image_uri.startswith("http://") or image_uri.startswith("https://"):
        response = requests.get(image_uri, stream=True)
        if response.status_code == 200:
            image_bytes = response.content
    else:
        image_bytes = open(image_uri, "rb").read()
    return image_bytes


class EmbeddingPredictionClient:
    """Wrapper around Prediction Service Client."""

    def __init__(
        self,
        project: str,
        location: str = "us-central1",
        api_regional_endpoint: str = "us-central1-aiplatform.googleapis.com",
    ):
        client_options = {"api_endpoint": api_regional_endpoint}
        # Initialize client that will be used to create and send requests.
        # This client only needs to be created once, and can be reused for multiple requests.
        self.client = aiplatform.gapic.PredictionServiceClient(
            client_options=client_options
        )
        self.location = location
        self.project = project

    def get_embedding(self, text: str = None, image_file: str = None, image_bytes = None):
        # Load image file
        if image_file:
            image_bytes = load_image_bytes(image_file)

        instance = struct_pb2.Struct()
        if text:
            instance.fields["text"].string_value = text

        if image_bytes:
            encoded_content = base64.b64encode(image_bytes).decode("utf-8")
            image_struct = instance.fields["image"].struct_value
            image_struct.fields["bytesBase64Encoded"].string_value = encoded_content

        instances = [instance]
        endpoint = (
            f"projects/{self.project}/locations/{self.location}"
            "/publishers/google/models/multimodalembedding@001"
        )
        response = self.client.predict(endpoint=endpoint, instances=instances)

        text_embedding = None
        if text:
            text_emb_value = response.predictions[0]["textEmbedding"]
            text_embedding = [v for v in text_emb_value]

        image_embedding = None
        if image_bytes:
            image_emb_value = response.predictions[0]["imageEmbedding"]
            image_embedding = [v for v in image_emb_value]

        return EmbeddingResponse(
            text_embedding=text_embedding, image_embedding=image_embedding
        )

@st.cache_resource
def get_embedding_client():
    return EmbeddingPredictionClient(project=PROJECT_ID)

client = get_embedding_client()

col1, col2 = st.columns(2)

txtQuery = col1.text_input("Search for images")
search_txt = col1.button("Search by text")

uploaded_file = col2.file_uploader("Choose a image", ['png', 'jpg'])
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    col2.image(bytes_data, "Source", 400)
search_image = col2.button("Search by image")

if search_txt:
    embedding = [client.get_embedding(text=txtQuery, image_file=None, image_bytes=None).text_embedding]
elif search_image:
    embedding = [client.get_embedding(text=None, image_file=None, image_bytes=bytes_data).image_embedding]
else:
    st.stop()

response = index_endpoint.find_neighbors(
    deployed_index_id=DEPLOYED_INDEX_ID,
    queries=embedding,
    num_neighbors=NUM_NEIGHBORS,
)

sorted_data = sorted(response[0], key=lambda x: x.distance, reverse=True)

image_directory = "extracted"
# Loop through the top max_images images and display them in the subplots

col = st.columns(5)
for i, response in enumerate(sorted_data):
    image_path = f"{image_directory}/{response.id}"
    score = response.distance

    # Display the image in the current subplot
    if os.path.exists(image_path):
        image = Image.open(image_path)

    col[i%5].image(image, f"{score}")