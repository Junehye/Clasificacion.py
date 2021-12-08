# -*- coding: utf-8 -*-
"""
UNIVERSIDAD AUTONOMA DEL ESTADO DE MEXICO
CU UAEM ZUMPANGO
UA: Redes neuronales
TEMA: Proyecto red neuronal con scikit learn
ALUMNOS: Dejaneyra June Salcedo Olivo
PROFESOR: Asdrubal Lopez Chau
DESCRIPCION: clasificacion 
@author: junes
"""
import pandas as pd
import numpy as np
#import librosa
from FuncSigmoid import dsigmoide
from sklearn.metrics import confusion_matrix

#audio_path = 'gato.wav'
audio = pd.read_csv('Sonidos.csv')
audio = audio.replace(np.nan,"0")
audio["Borrego"]=audio= ["Borrego"].replace ["Borregos"]
#print(type(x), type(sr))
# print(x.shape, sr)

#se coloca le nombre donde se coloco
#audio.to_csv('.csv',sep='\t')
#datos tipicos numeros
from sklearn.preprocessing import LabelbelEnconder#importa el metodo
enconder = LabelbelEnconder() #objeto
audio['borregos']= enconder.fit_transform(audio.Borrego.values)
#esta columna depende del archivo csv 
audio['gallo'] = enconder.fit_transform(audio.Gallo.values)
#print(juegos.borregos.unique())
X = audio [['borregos','gallo','gato']]
y = audio ['Genero']

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y)

from sklearn.preprocessing import StandarScaler
scaler = StandarScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_train)

#entrenamos red neuronal
#Este modelo optimiza la función de pérdida de registros utilizando LBFGS o descenso de gradiente estocástico.
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes = (10,10,10), max_iter = 500, alpha =0.0,
                    solver ='adam', random_state=21,tol=0.000000001)
#mpl = MLPClassifier(hidden_layer_sizes = (5,5,5,5),max_iter=5000) 
activation( 'relu')
 default='relu'
mlp.fit(X_train , y_train)
predictions = mpl.predict(X_test)
mtp.scatter(x_test, y_test, color="blue")
mtp.plot(x_train, x_pred, color="red")
mtp.xlabel("Gatos")
mtp.ylabel("Borregos")
mtp.show()

from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


