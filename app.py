from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np

def predict_disease(image_path, loaded_model, class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']):
    test_image = tf.keras.preprocessing.image.load_img(image_path, target_size=(256,256))
    test_image = tf.keras.preprocessing.image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    prediction = np.argmax(loaded_model.predict(test_image))
    return class_names[prediction]

def model_load():
    return tf.keras.models.load_model('potato_class_99.h5')

loaded_model = model_load()

# Initialize the app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        # print(file.filename)
        file.save(f"uploaded_images/{file.filename}")

        pred = predict_disease(f"uploaded_images/{file.filename}", loaded_model)

        return render_template('predict.html', value=pred)

if __name__ == '__main__':
    app.run(debug=True)