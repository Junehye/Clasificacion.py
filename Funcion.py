#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNIVERSIDAD AUTONOMA DEL ESTADO DE MEXICO
CU UAEM ZUMPANGO
UA: Redes neuronales
TEMA: Proyecto red neuronal con backpropagation
ALUMNOS: Dejaneyra June Salcedo Olivo
PROFESOR: asdruibal lopez chau
DESCRIPCION: Backpropagation 

@author: junes 
"""
import numpy as np
import random
import math

# función de sigmoide
def sigmoide(x):
    return 1.0/(1+np.exp(-x))

def dsigmoide(x):
    return 1.0 - x**2
