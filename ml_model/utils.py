
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA, IncrementalPCA
import tensorflow as tf

def get_prediction(arr) -> str:
    img_arr = process(arr)
    return predict(img_arr)

def process(arr):

    new = np.zeros(len(arr))
    # converting rgb to array to singular normalised value
    new = ((arr[:,0] << 16) + (arr[:,1] << 8) + arr[:,2]) / 0xffffff
    # conversion to matrix form
    mat = new.reshape(100, 100)


    # # using PCA to project data to lower dim
    k=20
    pca = PCA(n_components=k, svd_solver='randomized')
    img_comp = pca.fit_transform(mat)

    # converting to 2d form
    flat = img_comp.flatten()
    return flat

def predict(arr):
    m = tf.keras.models.load_model("ml_model/model.keras")
    pred = m.predict(arr[None])[0,0]
    if (pred > 0.5):
        return f"Benign {round(pred*100, 2)}% confidence"
    else:
        return f"Malignant {round(pred*100, 2)}% confidence"

