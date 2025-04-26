import pickle

# Load model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Misal inputan dummy
input_data = [[5.1, 3.5, 1.4, 0.2]]  # contoh input (sesuaikan)

# Predict
prediction = model.predict(input_data)

# Output hasil
print(prediction[0])
