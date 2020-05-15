
import streamlit as st
import json
import requests
import matplotlib.pyplot as plt
import numpy as np

URI = 'http://127.0.0.1:5000'
st.title('Neural Network Visualizer')

if st.button ('Get randomized prediction for MNIST data'):
    response = requests.post (URI, data = {})
    response = json.loads (response.text)
    preds = response.get('Prediction')
    image = response.get('Image')
    image = np.reshape (image, (28, 28))
    st.image (image, width = 150)
    st.markdown ('## Input image')
    
    for layer, p in enumerate (preds):
        numbers = np.squeeze(np.array(p))
        plt.figure(figsize=(32, 4))
        if layer == 2:
            row = 1
            col = 10
        else:
            row = 2
            col = 16
        for i, number in enumerate(numbers):
            plt.subplot (row, col, i+1)
            plt.imshow(number*np.ones((8, 8, 3)).astype('float32'))
            plt.xticks([])
            plt.yticks([])
            if layer == 2:
                plt.xlabel (str(i), fontsize=40)
        plt.subplots_adjust(wspace = 0.05, hspace = 0.05)
        plt.tight_layout()
        st.text ('Layer {}'.format(layer+1))
        st.pyplot()
