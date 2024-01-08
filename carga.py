
from tensorflow.keras.datasets import mnist
import math, time
import matplotlib.pyplot as plt
import numpy as np
#!pip install seaborn
import seaborn as sns
#%matplotlib inline
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from PIL import Image

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from imblearn.metrics import specificity_score
from matplotlib import*
from matplotlib.cm import register_cmap
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
#from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.models import model_from_json
from keras.models import load_model

from sklearn.svm import SVC #SVR para regresión
from sklearn.metrics import classification_report
from keras import models
from keras.layers import BatchNormalization, MaxPool2D, GlobalMaxPool2D
#Arquitecturas de Transfer learning. Puedes configurar parámetros específicos de cada arquitectura
from tensorflow.keras.applications.vgg16 import VGG16
#from keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Dense, concatenate, Dropout
from tensorflow.keras.applications import VGG16, Xception
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D

print('Módulos importados')


# Funciones para guardar y cargar objetos pickle
def guardarObjeto(pipeline,nombreArchivo):
    print("Guardando Objeto en Archivo")
    with open(nombreArchivo+'.pickle', 'wb') as handle:
        pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Objeto Guardado en Archivo")
def cargarObjeto(nombreArchivo):
    with open(nombreArchivo+'.pickle', 'rb') as handle:
        pipeline = pickle.load(handle)
        print("Objeto Cargado desde Archivo")
    return pipeline
# Funciones para guardar y cargar la Red Neuronal (Arquitectura y Pesos)
def guardarNN(model,nombreArchivo):
    print("Guardando Red Neuronal en Archivo")
    model.save(nombreArchivo+'.h5')
    print("Red Neuronal Guardada en Archivo")

def cargarNN(nombreArchivo):
    model = load_model(nombreArchivo+'.h5')
    print("Red Neuronal Cargada desde Archivo")
    return model

# Función para medir la calidad de modelos
def obtenerResultados(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    specificity = specificity_score(y_test, y_pred, average='macro')

    accuracy=str(round(accuracy, 4))
    precision=str(round(precision, 4))
    recall=str(round(recall, 4))
    f1=str(round(f1, 4))
    specificity=str(round(specificity, 4))

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall o Sensitivity:", recall)
    print("F1-Score:", f1)
    print("Specificity:", specificity)

    # Se evalúa el modelo con otras medidas de calidad y se presenta la matriz de confusión
    MAE=metrics.mean_absolute_error(y_test, y_pred)
    MSE=metrics.mean_squared_error(y_test, y_pred)
    RMSE=np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    MAE=str(round(MAE, 4))
    MSE=str(round(MSE, 4))
    RMSE=str(round(RMSE, 4))

    print('Mean Absolute Error (MAE):', MAE)
    print('Mean Squared Error (MSE):', MSE)
    print('Root Mean Squared Error (RMSE):', RMSE)

    #plt.figure(figsize=(15,10))
    fx=sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, fmt=".2f",cmap="GnBu")
    fx.set_title('Confusion Matrix \n');
    fx.set_xlabel('\n Valores de predicción\n')
    fx.set_ylabel('Valores reales\n');
    #fx.xaxis.set_ticklabels(_load_label_names())
    #fx.yaxis.set_ticklabels(_load_label_names())
    plt.show()

    #return accuracy, precision, recall, f1, specificity, MAE, MSE, RMSE
print('Funciones para guardar y cargar modelos personalizados')


# Rutas de los archivos 

ruta_imagenes = 'imagenes.pkl'
ruta_etiquetas = 'etiquetas.pkl'


# Cargar el archivo de imágenes
with open(ruta_imagenes, 'rb') as file_imagenes:
    imagenes_cargadas = pickle.load(file_imagenes)

# Cargar el archivo de etiquetas
with open(ruta_etiquetas, 'rb') as file_etiquetas:
    etiquetas_cargadas = pickle.load(file_etiquetas)


print(etiquetas_cargadas)

# Supongamos que tienes un array de etiquetas llamado etiquetas_cargadas
# Puedes usar np.unique para obtener los valores únicos
valores_unicos = np.unique(etiquetas_cargadas)

# Luego, puedes crear el mapeo basado en los valores únicos
mapping = {valor: indice for indice, valor in enumerate(valores_unicos)}

# Ahora puedes aplicar el mapeo a las etiquetas
etiquetas_cargadas = np.vectorize(lambda x: mapping[x])(etiquetas_cargadas)

# Imprimir los valores únicos y las etiquetas mapeadas
print("Valores únicos en las etiquetas:", valores_unicos)
print("Mapeo de valores:", mapping)
print("Etiquetas mapeadas:", etiquetas_cargadas)

# Convertir la lista de imágenes a un array de NumPy
X_tr = np.array(imagenes_cargadas)

# Mostrar las dimensiones del array
print("X_train shape", X_tr.shape)

# Convertir a tipo de datos uint8
X_tr = X_tr.astype(np.uint8)

# Resto del código
y_tr = etiquetas_cargadas
print("y_train shape", y_tr.shape)

from sklearn.model_selection import train_test_split

# SPLIT EN TRAIN Y TEST
X_train, X_test, y_train, y_test = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr)

# Imprimir las formas de los conjuntos de entrenamiento y prueba
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Normalizar los datos dividiendo por 255 ---Coloque esto para hacer la parte de 3 canales
#X_train = X_train / 255.0
#X_test = X_test / 255.0

# Imprimir mensaje indicando que los datos están normalizados
print('Datos normalizados')

import cv2
import numpy as np

# Supongamos que X_train tiene la forma (número de imágenes, altura, ancho)
# y X_test tiene la forma (número de imágenes, altura, ancho)

# Redimensionar a 224x224
X_train_resized = np.array([cv2.resize(img, (224, 224)) for img in X_train])
X_test_resized = np.array([cv2.resize(img, (224, 224)) for img in X_test])

# Replicar el canal único tres veces para obtener tres canales idénticos
X_train = np.repeat(X_train_resized[..., np.newaxis], 3, axis=-1)
X_test = np.repeat(X_test_resized[..., np.newaxis], 3, axis=-1)

# Normalizar los datos dividiendo por 255
X_train = X_train / 255.0
X_test = X_test / 255.0

# Ahora, X_train tiene la forma (número de imágenes, 224, 224, 3)
print("X_train shape:", X_train.shape)

# Ahora, X_test tiene la forma (número de imágenes, 224, 224, 3)
print("X_test shape:", X_test.shape)



# FASE EXPERIMENTAL


import tensorflow as tf
from keras.applications import EfficientNetB4
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

# Convertir etiquetas a one-hot encoding
num_classes = 8  # Número real de clases
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)
print('Transformación de salida a binario')

# Dimensiones de entrada de tu modelo
input_shape = (224, 224, 3)  # Imágenes en formato RGB

# Cargar el modelo preentrenado EfficientNetB4
base_model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=input_shape)

# Descongelar solo las últimas 10 capas del modelo base
base_model.trainable = True
for layer in base_model.layers[:-10]:  # Deja las últimas 10 capas congeladas
    layer.trainable = False

# Inicializar tu modelo
classifierCNN = Sequential()

# Agregar el modelo base (EfficientNetB4) al modelo secuencial
classifierCNN.add(base_model)

# Agregar capas adicionales
classifierCNN.add(GlobalAveragePooling2D())
classifierCNN.add(Dense(32, activation='relu'))
classifierCNN.add(Dense(num_classes, activation='softmax'))

# Compilar el modelo
classifierCNN.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

# Crear un generador de imágenes con aumentos de datos
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Entrenar el modelo con el generador de imágenes
batch_size = 32
epochs = 10

h = classifierCNN.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(X_train) // batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test)
)

classifierCNN.summary()

batch_size = 50
epochs = 10

h = classifierCNN.fit(X_train, y_train, batch_size=batch_size,epochs=epochs, validation_data=(X_test, y_test))
acc_cnn=str(round(test_acc_cnn[1], 4))
print('\nCNN Accuracy: ',acc_cnn)
error_rate_cnn=str(round(test_acc_cnn[0], 4))
print('\nCNN Loss: ',acc_cnn)