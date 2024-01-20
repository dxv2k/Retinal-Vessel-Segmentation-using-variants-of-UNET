import onnx
import streamlit as st
from PIL import Image
import numpy as np

from demo.predict import (
    unet_onnx_predict,
    res_unet_onnx_predict,
    attn_unet_onnx_predict,
    res_attn_unet_onnx_predict,
)

unet_onnx = onnx.load("./onnx_models/unet.onnx")
res_unet_onnx = onnx.load("./onnx_models/res_unet.onnx")
attn_unet_onnx = onnx.load("./onnx_models/attn_unet.onnx")
res_attn_unet_onnx = onnx.load("./onnx_models/res_attn_unet.onnx")


# Define your prediction functions (replace with actual implementation)
def unet_pred(image: np.ndarray) -> np.ndarray:
    global unet_onnx
    result = unet_onnx_predict(onnx_model=unet_onnx, input_image=image)
    return result


def res_unet_pred(image: np.ndarray) -> np.ndarray:
    global res_unet_onnx
    result = res_unet_onnx_predict(onnx_model=res_unet_onnx, input_image=image)
    return result


def attn_unet_pred(image: np.ndarray) -> np.ndarray:
    global attn_unet_onnx
    result = attn_unet_onnx_predict(onnx_model=attn_unet_onnx, input_image=image)
    return result


def res_attn_unet_pred(image: np.ndarray) -> np.ndarray:
    global res_attn_unet_onnx
    result = res_attn_unet_onnx_predict(
        onnx_model=res_attn_unet_onnx, input_image=image
    )
    return result


# Main Streamlit app
def main():
    st.title("Blood Vessel Segmentation")

    # Create two big columns
    col1, col2 = st.columns((2, 4))

    # File uploader in the first column
    uploaded_file = col1.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"]
    )

    # Display the uploaded image and prediction button in the first column
    if uploaded_file is not None:
        col1.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Perform predictions when the user clicks the button
        if col1.button("Make Predictions"):
            # Convert the uploaded file to PIL Image
            pil_image = Image.open(uploaded_file)

            # Call prediction functions
            prediction1 = unet_pred(np.array(pil_image))
            prediction2 = res_unet_pred(np.array(pil_image))
            prediction3 = attn_unet_pred(np.array(pil_image))
            prediction4 = res_attn_unet_pred(np.array(pil_image))

            # Normalize predictions to [0.0, 1.0] range
            prediction1 = (prediction1 - prediction1.min()) / (
                prediction1.max() - prediction1.min()
            )
            prediction2 = (prediction2 - prediction2.min()) / (
                prediction2.max() - prediction2.min()
            )
            prediction3 = (prediction3 - prediction3.min()) / (
                prediction3.max() - prediction3.min()
            )
            prediction4 = (prediction3 - prediction3.min()) / (
                prediction3.max() - prediction3.min()
            )
            # Display the predictions as images in the second column
            col2.image(prediction1, caption="U-Net", use_column_width=True)
            col2.image(prediction2, caption="ResU-Net", use_column_width=True)
            col2.image(prediction3, caption="Attention U-Net", use_column_width=True)
            col2.image(prediction4, caption="ResAttention U-Net", use_column_width=True)


if __name__ == "__main__":
    main()
