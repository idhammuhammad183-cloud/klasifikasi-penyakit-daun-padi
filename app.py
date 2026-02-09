import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Klasifikasi Penyakit Daun Padi",
    layout="centered"
)

st.title("🌾 Klasifikasi Penyakit Daun Padi")
st.write("Website klasifikasi penyakit daun padi berbasis CNN")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_penyakit_daun_padi.h5")

model = load_model()

class_names = [
    "Bacterial leaf blight",
    "Brown spot",
    "Leaf smut"
]

uploaded_file = st.file_uploader(
    "Upload gambar daun padi",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar daun padi", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    confidence = np.max(predictions)
    predicted_class = class_names[np.argmax(predictions)]

    st.subheader("Hasil Klasifikasi")
    st.write(f"**Penyakit :** {predicted_class}")
    st.write(f"**Confidence :** {confidence * 100:.2f}%")
