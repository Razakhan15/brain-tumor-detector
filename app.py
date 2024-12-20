import os
import tensorflow as tensorflow
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
model = load_model('BrainTumor10EpochsCategorical.h5')
app.config['CORS_HEADERS'] = 'Content-Type'

def get_className(classNo):
    if classNo == 0:
        return "Tumor not detected"
    elif classNo == 1:
        return "Tumor detected"

def getResult(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64,64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis = 0)
    result = model.predict(input_img)
    result_final= np.argmax(result,axis=1)
    return result_final

@app.route('/', methods=['GET'])
@cross_origin()
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(  
            basepath, 'uploads', secure_filename(f.filename)
        )
        f.save(file_path)
        value = getResult(file_path)
        result = get_className(value)
        return result
    return None

if __name__ == '__main__':
    app.run()