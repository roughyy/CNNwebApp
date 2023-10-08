from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.preprocessing import image
import os
import tensorflow as tf
import psycopg2
from dotenv import load_dotenv
import numpy as np
from PIL import Image
load_dotenv()

app = Flask(__name__)

db_connection = psycopg2.connect(
    host = os.getenv('PSQL_HOST'),
    database = os.getenv('PSQL_DATABASE'),
    user = os.getenv('PSQL_USER'),
    password = os.getenv('PSQL_PASSWORD'),
    port = os.getenv('PSQL_PORT')
)

model = tf.keras.models.load_model('./models/RPS.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods = ['POST'])
def upload():
    if 'file' in request.files:
        file = request.files['file']
        img = Image.open(file.stream)
        img = img.resize((150,150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction[0])
        cur = db_connection.cursor()
        cur.execute("INSERT INTO image_classification (image, value) VALUES (%s, %s) RETURNING id", (psycopg2.Binary(file.read()), predicted_class))
        inserted_id = cur.fetchone()[0]
        db_connection.commit()
        cur.close()
        return redirect(url_for('result', num=inserted_id))
    return render_template('index.html')

@app.route('/result', methods=['GET'])
def result():
    num = request.args.get('id')
    cursor = db_connection.cursor()
    cursor.execute('SELECT * FROM image_classification where id = %s',(num))
    hasil = cursor.fetchone()
    cursor.close()
    
    return render_template('result.html', result=hasil)

