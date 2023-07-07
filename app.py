import streamlit as st
from skimage import feature, transform, io
from pickle import load
import numpy as np
import cv2


def preprocessing_1(photo_dir):
    
    
    features = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
    angle = [0, 45, 90, 135, 180]

    res = np.array([])
    
    for j in range(len(features)):
        
        a = feature.graycomatrix(photo_dir, distances=[2], angles=angle, levels=256,
                            symmetric=True, normed=True)
        a = feature.graycoprops(a, prop=features[j]).flatten()

        for k in range(len(angle)):

            res = np.append(a[k], res)
    
    return res

model = load(open('model.pkl', 'rb'))
scale = load(open('scaling.pkl', 'rb'))
label = load(open('label.pkl', 'rb'))

def main():

    st.title('Image Prediction dengan Neural Network')
    st.header('Prediksi ekspresi gambar apakah ekspresi senang atau sedih disini : ')
    uploaded_file = st.file_uploader("Upload Image")

    if uploaded_file is not None:

        image = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (48, 48))
        image = preprocessing_1(image)
        res = model.predict(np.array(scale.transform([image])))
    
        if res[:, 0] > res[:, 1]:
            st.header('Senang')
        else:
            st.header('Sedih')

if __name__ == '__main__':
    main()

