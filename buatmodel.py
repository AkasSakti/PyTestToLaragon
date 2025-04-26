import pickle
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Load dataset contoh (Iris dataset)
iris = load_iris()
X = iris.data
y = iris.target

# Buat model sederhana
model = DecisionTreeClassifier()
model.fit(X, y)

# Simpan model ke file model.pkl
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model berhasil disimpan ke model.pkl")
