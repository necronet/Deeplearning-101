import os
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.exceptions import NotFound

print("the app is running")
model = keras.models.load_model('./model/image_classification.h5')
UPLOAD_FOLDER = os.path.join("/uploads/")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/classify', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('score_image', filename = filename))

    return render_template('dog_cat_classification.html')

@app.route('/uploads/<filename>')
def uploaded_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/score/<filename>')
def score_image(filename):
    image_size = (180, 180)
    testImagePath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    img = keras.preprocessing.image.load_img(testImagePath, target_size=image_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 

    predictions = model.predict(img_array)
    score = predictions[0]
    print(
        "This image is %.2f percent cat and %.2f percent dog."
        % (100 * (1 - score), 100 * score)
    )

    return render_template('score_image.html',  cat_score = 100 * (1 - score), dog_score = 100 * score, filename = filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
