from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

with open('app/model/iris_model.pkl', 'rb') as file:
    model = pickle.load(file)

class_names = ['Setosa', 'Versicolor', 'Virginica']

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
        predicted_class = class_names[prediction[0]]
        
        # Trả về kết quả dự đoán và không lưu kết quả nào trong session hoặc cache
        return render_template('index.html', prediction_text=f'Loài hoa dự đoán là: {predicted_class}')

if __name__ == '__main__':
    app.run(debug=True)
