import cv2
import onnx
import time
import datetime
import numpy as np
import skimage
from onnxruntime import InferenceSession
from patchify import patchify, unpatchify


# CLAHE
def clahe_equalized(imgs):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgs_equalized = clahe.apply(imgs)
    return imgs_equalized


def unet_onnx_predict(
    onnx_model: onnx.onnx_ml_pb2.ModelProto,
    # filepath: str | None = None,
    input_image: np.ndarray,
    patch_size: int = 512,
) -> np.ndarray:
    # _img = skimage.io.imread(filepath)  # test image
    _img = input_image

    predicted_patches = []
    start = time.time()

    test = clahe_equalized(_img)  # applying CLAHE
    SIZE_X = (
        _img.shape[1] // patch_size
    ) * patch_size  # getting size multiple of patch size
    SIZE_Y = (
        _img.shape[0] // patch_size
    ) * patch_size  # getting size multiple of patch size
    test = cv2.resize(test, (SIZE_X, SIZE_Y))
    test = np.array(test)
    patches = patchify(
        test, (patch_size, patch_size), step=patch_size
    )  # create patches(patch_sizexpatch_sizex1)
    print(f"Number of patches: {len(patches)}")

    reconstructed_image = None
    sess = InferenceSession(onnx_model.SerializeToString())
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            single_patch = patches[i, j, :, :]
            single_patch_norm = (single_patch.astype("float32")) / 255.0
            single_patch_norm = np.expand_dims(np.array(single_patch_norm), axis=-1)
            single_patch_input = np.expand_dims(single_patch_norm, 0)
            result = sess.run(None, {"input_1": single_patch_input})
            result = (result[0][0, :, :, 0] > 0.5).astype(np.uint8)
            predicted_patches.append(result)
            predicted_patches = np.array(predicted_patches)
    predicted_patches_reshaped = np.reshape(
        predicted_patches,
        (patches.shape[0], patches.shape[1], patch_size, patch_size),
    )

    # join patches to form whole img
    reconstructed_image: np.ndarray = unpatchify(predicted_patches_reshaped, test.shape)

    stop = time.time()
    print("Execution time: ", (stop - start))  # computation time

    return reconstructed_image



def res_unet_onnx_predict(
    onnx_model: onnx.onnx_ml_pb2.ModelProto,
    # filepath: str | None = None,
    input_image: np.ndarray,
    patch_size: int = 512,
) -> np.ndarray:
    # _img = skimage.io.imread(filepath)  # test image
    _img = input_image

    predicted_patches = []
    start = time.time()

    test = clahe_equalized(_img)  # applying CLAHE
    SIZE_X = (
        _img.shape[1] // patch_size
    ) * patch_size  # getting size multiple of patch size
    SIZE_Y = (
        _img.shape[0] // patch_size
    ) * patch_size  # getting size multiple of patch size
    test = cv2.resize(test, (SIZE_X, SIZE_Y))
    test = np.array(test)
    patches = patchify(
        test, (patch_size, patch_size), step=patch_size
    )  # create patches(patch_sizexpatch_sizex1)
    print(f"Number of patches: {len(patches)}")

    reconstructed_image = None
    sess = InferenceSession(onnx_model.SerializeToString())
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            single_patch = patches[i, j, :, :]
            single_patch_norm = (single_patch.astype("float32")) / 255.0
            single_patch_norm = np.expand_dims(np.array(single_patch_norm), axis=-1)
            single_patch_input = np.expand_dims(single_patch_norm, 0)
            result = sess.run(None, {"input_2": single_patch_input})
            result = (result[0][0, :, :, 0] > 0.5).astype(np.uint8)
            predicted_patches.append(result)
            predicted_patches = np.array(predicted_patches)
    predicted_patches_reshaped = np.reshape(
        predicted_patches,
        (patches.shape[0], patches.shape[1], patch_size, patch_size),
    )

    # join patches to form whole img
    reconstructed_image: np.ndarray = unpatchify(predicted_patches_reshaped, test.shape)

    stop = time.time()
    print("Execution time: ", (stop - start))  # computation time

    return reconstructed_image



def attn_unet_onnx_predict(
    onnx_model: onnx.onnx_ml_pb2.ModelProto,
    # filepath: str | None = None,
    input_image: np.ndarray,
    patch_size: int = 512,
) -> np.ndarray:
    # _img = skimage.io.imread(filepath)  # test image
    _img = input_image

    predicted_patches = []
    start = time.time()

    test = clahe_equalized(_img)  # applying CLAHE
    SIZE_X = (
        _img.shape[1] // patch_size
    ) * patch_size  # getting size multiple of patch size
    SIZE_Y = (
        _img.shape[0] // patch_size
    ) * patch_size  # getting size multiple of patch size
    test = cv2.resize(test, (SIZE_X, SIZE_Y))
    test = np.array(test)
    patches = patchify(
        test, (patch_size, patch_size), step=patch_size
    )  # create patches(patch_sizexpatch_sizex1)
    print(f"Number of patches: {len(patches)}")

    reconstructed_image = None
    sess = InferenceSession(onnx_model.SerializeToString())
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            single_patch = patches[i, j, :, :]
            single_patch_norm = (single_patch.astype("float32")) / 255.0
            single_patch_norm = np.expand_dims(np.array(single_patch_norm), axis=-1)
            single_patch_input = np.expand_dims(single_patch_norm, 0)
            result = sess.run(None, {"input_3": single_patch_input})
            result = (result[0][0, :, :, 0] > 0.5).astype(np.uint8)
            predicted_patches.append(result)
            predicted_patches = np.array(predicted_patches)
    predicted_patches_reshaped = np.reshape(
        predicted_patches,
        (patches.shape[0], patches.shape[1], patch_size, patch_size),
    )

    # join patches to form whole img
    reconstructed_image: np.ndarray = unpatchify(predicted_patches_reshaped, test.shape)

    stop = time.time()
    print("Execution time: ", (stop - start))  # computation time

    return reconstructed_image

def res_attn_unet_onnx_predict(
    onnx_model: onnx.onnx_ml_pb2.ModelProto,
    # filepath: str | None = None,
    input_image: np.ndarray,
    patch_size: int = 512,
) -> np.ndarray:
    # _img = skimage.io.imread(filepath)  # test image
    _img = input_image

    predicted_patches = []
    start = time.time()

    test = clahe_equalized(_img)  # applying CLAHE
    SIZE_X = (
        _img.shape[1] // patch_size
    ) * patch_size  # getting size multiple of patch size
    SIZE_Y = (
        _img.shape[0] // patch_size
    ) * patch_size  # getting size multiple of patch size
    test = cv2.resize(test, (SIZE_X, SIZE_Y))
    test = np.array(test)
    patches = patchify(
        test, (patch_size, patch_size), step=patch_size
    )  # create patches(patch_sizexpatch_sizex1)
    print(f"Number of patches: {len(patches)}")

    reconstructed_image = None
    sess = InferenceSession(onnx_model.SerializeToString())
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            single_patch = patches[i, j, :, :]
            single_patch_norm = (single_patch.astype("float32")) / 255.0
            single_patch_norm = np.expand_dims(np.array(single_patch_norm), axis=-1)
            single_patch_input = np.expand_dims(single_patch_norm, 0)
            result = sess.run(None, {"input_4": single_patch_input})
            result = (result[0][0, :, :, 0] > 0.5).astype(np.uint8)
            predicted_patches.append(result)
            predicted_patches = np.array(predicted_patches)
    predicted_patches_reshaped = np.reshape(
        predicted_patches,
        (patches.shape[0], patches.shape[1], patch_size, patch_size),
    )

    # join patches to form whole img
    reconstructed_image: np.ndarray = unpatchify(predicted_patches_reshaped, test.shape)

    stop = time.time()
    print("Execution time: ", (stop - start))  # computation time

    return reconstructed_image




if __name__ == "__main__":
    unet_model = onnx.load("../onnx_models/unet.onnx")
    res_unet_model = onnx.load("../onnx_models/res_unet.onnx")
    attn_unet_model = onnx.load("../onnx_models/attn_unet.onnx")
    res_attn_unet_model = onnx.load("../onnx_models/res_attn_unet.onnx")

    image = "../data/test.jpg"
    image = skimage.io.imread(image)  # test image

    unet_result = unet_onnx_predict(unet_model, image)
    res_unet_result = res_unet_onnx_predict(res_unet_model, image)
    attn_unet_result = attn_unet_onnx_predict(attn_unet_model, image)
    res_attn_unet_result = res_attn_unet_onnx_predict(res_attn_unet_model, image)

    print("unet: ",unet_result.shape)
    print("res_unet: ",res_unet_result.shape)
    print("attn_unet: ",attn_unet_result.shape)
    print("res_attn_unet: ",res_attn_unet_result.shape)

