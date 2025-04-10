import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Charger les données
data = pd.read_csv('train.csv')
data.head()

# Séparer les variables dépendantes et indépendantes
x = data.drop(['label'], axis=1).values
y = data['label']
print(x.shape)  # (42000, 784)

# Visualiser quelques échantillons
plt.figure(figsize=(10, 8))
for i in np.arange(1, 10):
    plt.subplot(int('33' + str(i)))
    plt.imshow(x[i + 10].reshape(28, 28), cmap=plt.cm.binary)
    plt.title(y[i + 10])
    plt.axis('off')
plt.show()

# Diviser les données en ensembles d'entraînement et de test
num_train = 38000
x_train, x_test, y_train, y_test = x[:num_train], x[num_train:], y[:num_train], y[num_train:]

# Implémentation de l'algorithme KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

# Prédictions sur l'ensemble de test
gues = knn.predict(x_test)

# Évaluer la précision du modèle KNN
from sklearn.metrics import accuracy_score, confusion_matrix
print(accuracy_score(y_test, gues))  # Précision obtenue
print(confusion_matrix(y_test, gues))

# Implémentation d'un réseau de neurones artificiel
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

# Préparer les étiquettes pour la classification catégorielle
y_train = to_categorical(y_train)

# Construire le modèle
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compiler le modèle
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Entraîner le modèle
model.fit(x_train, y_train, epochs=200, verbose=2)

# Préparer les étiquettes de test et évaluer le modèle
y_test = to_categorical(y_test)
model.evaluate(x_test, y_test)
