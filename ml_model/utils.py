
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA, IncrementalPCA
import tensorflow as tf

def get_prediction(image) -> str:
    img_arr = process(image)
    return predict(img_arr)

def process(image):
    img = Image.open(image).resize((100,100))
    arr = np.array(img.getdata())
    new = np.zeros(len(arr))
    # converting rgb to array to singular normalised value
    new = ((arr[:,0] << 16) + (arr[:,1] << 8) + arr[:,2]) / 0xffffff
    # conversion to matrix form
    mat = new.reshape(img.size[0], img.size[1])


    # # using PCA to project data to lower dim
    k=20
    pca = PCA(n_components=k, svd_solver='randomized')
    img_comp = pca.fit_transform(mat)

    # converting to 2d form
    flat = img_comp.flatten()
    return flat

def predict(arr):
    m = tf.keras.models.load_model("model.keras")
    pred = m.predict(arr[None])
    if (pred > 0.5):
        return "Benign"
    else:
        return "Malignant"

