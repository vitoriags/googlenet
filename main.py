import tensorflow as tf
from tensorflow import keras
from glob import glob
import os
import time
import cv2
import pandas as pd
import numpy as np
import random
from funcoes import *

# diretório atual
diretorio = os.getcwd()
print(diretorio)

# executado apenas uma vez
criarDiretorio(diretorio, 'train', '0')
criarDiretorio(diretorio, 'train', '1')
criarDiretorio(diretorio, 'validation', '0')
criarDiretorio(diretorio, 'validation', '1')


# abrindo tabela de rótulos
gx1 = pd.read_csv("gz2_hart16.csv", sep=',')

# abrindo tabela do kaggle
gx2 = pd.read_csv("gz2_filename_mapping.csv", sep=',')

# agrupando através do inner join
gx = gx1.merge(gx2, how='inner')

# filtrando informações que iremos utilizar
colunas = ['dr7objid', 'gz2_class', 'asset_id']
gx = gx[colunas]

# extraindo caminhos de todas as imagens
img_end = glob(os.path.join('images','*.jpg'))

# gerarando dataframe com os caminhos
im = pd.DataFrame(img_end, columns=["images"])
print(im)

# corrigindo os caminhos
im["images"] = im["images"].str.replace('\\', '/')

# gerando caminho das imagens com o asset_id para comparação com imagens[images]
gx['images'] = gx['asset_id'].map(lambda num: "images/" + str(num) + ".jpg")

# agrupando os caminhos das imagens no dataset completo
gx = pd.merge(im, gx, how='inner', on='images')

# removendo imagens de estrelas
gx = gx[gx["gz2_class"] != "A"]

# criando uma nova coluna com a classificação binária de elipcas [1] e espirais [0]
gx['newclass'] = gx['gz2_class'].map(lambda num: 1 if 'E' in num else 0)

# gx.to_csv("tabelafinal.csv")
# gx = pd.read_csv("tabelafinal.csv", sep=',')

# carregando lista do caminho das imagens
galaxias = list(gx['images'])

# selecionando as linhas onde 'newclass' é igual a 0 e guardando as informações de 'images' em uma lista
esp = gx.query('newclass == 0')
spirals = list(esp['images'])
print(spirals[:20])

# randomizando vetor de caminhos das imagens
random.shuffle(galaxias)

# leitura/resize/gravação em diretórios separados
cont = 0
for nome in galaxias:
    img = cv2.imread(nome)
    img = cv2.resize(img, dsize=(227, 227), interpolation=cv2.INTER_CUBIC)

    if cont < len(galaxias) * 0.9:
        if nome in spirals:
            print(f'{cont} | {nome} | treino - tipo: 0')
            cv2.imwrite(os.path.join(f'{diretorio}/train/0', nome[7:]), img)
            cont += 1
        else:
            print(f'{cont} | {nome} | treino - tipo: 1')
            cv2.imwrite(os.path.join(f'{diretorio}/train/1', nome[7:]), img)
            cont += 1

    else:
        if nome in spirals:
            print(f'{cont} | {nome} | validação - tipo: 0')
            cv2.imwrite(os.path.join(f'{diretorio}/validation/0', nome[7:]), img)
            cont += 1
        else:
            print(f'{cont} | {nome} | validação - tipo: 1')
            cv2.imwrite(os.path.join(f'{diretorio}/validation/1', nome[7:]), img)
            cont += 1
