import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import joblib

# Generamos datos de ejemplo (ingresos, gastos, ahorro)
data = np.array([
    [2000, 1500, 500], 
    [3000, 2000, 1000], 
    [4000, 2500, 1500], 
    [5000, 3500, 1500],
    [6000, 4000, 2000]
])

# Normalizar los datos
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Separar características (X) y etiquetas (Y)
X = data_scaled[:, :2]  # ingresos y gastos
Y = data_scaled[:, 2:]  # ahorro

# Crear el modelo
model = keras.Sequential([
    layers.Dense(10, activation="relu"),
    layers.Dense(5, activation="relu"),
    layers.Dense(1, activation="linear")  # Predicción del ahorro
])

# Compilar el modelo
model.compile(optimizer="adam", loss="mse")

# Entrenar el modelo
model.fit(X, Y, epochs=500, verbose=1)

# Guardar el modelo entrenado
model.save("ai_financial_core.h5")

# Guardar el escalador para normalizar futuros datos
joblib.dump(scaler, "scaler.pkl")

print("✅ Modelo entrenado y guardado correctamente.")
