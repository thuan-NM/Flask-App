
from flask import Flask,render_template,url_for,request
from flask_material import Material
import numpy as np
import pickle

app = Flask(__name__)
Material(app)

with open('model/iris_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Lấy dữ liệu từ form
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        
        # Dự đoán
        prediction = model.predict(final_features)
        
        # Kết quả
        output = prediction[0]
        
        return render_template('index.html', prediction_text=f'Loài hoa dự đoán là: {output}')

if __name__ == '__main__':
	app.run(debug=True)
