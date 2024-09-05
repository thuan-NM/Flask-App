# mssv_train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Đọc dữ liệu từ file iris.csv
data = pd.read_csv('app/model/iris.csv')

# Tiền xử lý dữ liệu (chuyển đổi nhãn từ chuỗi thành số)
label_encoder = LabelEncoder()
data['variety'] = label_encoder.fit_transform(data['variety'])

# Tách dữ liệu thành features và labels
X = data.drop('variety', axis=1)
y = data['variety']

# Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Huấn luyện mô hình (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Lưu mô hình
with open('app/model/iris_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Mô hình đã được huấn luyện và lưu.")
