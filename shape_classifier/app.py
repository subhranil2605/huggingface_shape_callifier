import gradio as gr

import tensorflow as tf
import numpy as np


MODEL = tf.keras.models.load_model("my_model")
CLASS_NAMES = ['circle', 'rectangle']


def classify_predict(inp):
    image = inp
    image_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(image_batch)
    values, indices = tf.math.top_k(predictions, 2)
    predicted_values = values.numpy().tolist()[0]
    indcs = indices.numpy().tolist()[0]
    confidences = {CLASS_NAMES[i]: round(v, 4) for i, v in zip(indcs, predicted_values)}
    print(confidences)
    return confidences


interface = gr.Interface(
    fn=classify_predict,
    inputs=gr.inputs.Image(shape=(64, 64)),
    outputs="label",
    examples=["28.jpg", "57.jpg"]
)

interface.launch()
