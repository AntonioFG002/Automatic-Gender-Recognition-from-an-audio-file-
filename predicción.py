import preprocesamiento as prep
import modelo
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Obtener espectrogramas, etiquetas y nombres de archivos
Spectro = prep.getSpectrogram()
Label = prep.getLabels()
file_names = prep.getFileNames()

# Dividir los datos en conjuntos de entrenamiento y prueba manualmente
num_samples = Spectro.shape[0]
indices = np.arange(num_samples)
np.random.shuffle(indices) 

# Definir el tamaño del conjunto de prueba
test_size = int(0.2 * num_samples)

test_indices = indices[:test_size]
train_indices = indices[test_size:]

# Crear los conjuntos de entrenamiento y prueba
Spectro_train, Spectro_test = Spectro[train_indices], Spectro[test_indices]
Label_train, Label_test = Label[train_indices], Label[test_indices]
file_names_test = [file_names[i] for i in test_indices]  

input_shape = Spectro_train.shape[1:]

# Construir el modelo
model = modelo.build_cnn_model(input_shape)
model.summary()

# Crear un generador de datos con aumento de datos
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

datagen.fit(Spectro_train)

# Entrenar el modelo con el generador de datos y Early Stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(
    datagen.flow(Spectro_train, Label_train, batch_size=32),
    epochs=50,
    validation_data=(Spectro_test, Label_test),
    callbacks=[early_stopping]
)

# Realizar la predicción
predictions = model.predict(Spectro_test)
predictions_binary = (predictions > 0.5).astype(int).flatten()  

# Calcular el accuracy
correct_predictions = (predictions_binary == Label_test).sum()
total_predictions = len(Label_test)
accuracy = correct_predictions / total_predictions
print(f"Accuracy total en el conjunto de prueba: {accuracy * 100:.2f}%")

# Calcular métricas adicionales: Recall y F1-Score
TP = np.sum((predictions_binary == 1) & (Label_test == 1))  
FP = np.sum((predictions_binary == 1) & (Label_test == 0))  
FN = np.sum((predictions_binary == 0) & (Label_test == 1))  

# Evitar división por cero en el cálculo de recall y F1-score
if TP + FN > 0:
    recall = TP / (TP + FN)
else:
    recall = 0

if (2 * TP + FP + FN) > 0:
    f1_score = 2 * TP / (2 * TP + FP + FN)
else:
    f1_score = 0

print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1_score:.2f}")

# Interpretar el resultado para cada predicción
for i, prediction in enumerate(predictions):
    audio_name = file_names_test[i]
    if prediction > 0.5:
        print(f"{audio_name}: El hablante es femenino")
    else:
        print(f"{audio_name}: El hablante es masculino")
