import tensorflow as tf
from tensorflow import keras
from glob import glob
import os
import time
import cv2
import pandas as pd
import numpy as np
import random
import zipfile
import sys
import shutil
from funcoes import *

# diretório atual
diretorio = os.getcwd()

## excluir as pastas ---- executado apenas uma vez
# shutil.rmtree('test')
# shutil.rmtree('train')
# shutil.rmtree('validation')
# # shutil.rmtree('logs-fit')

# abrindo tabela de rótulos
gx1 = pd.read_csv("/content/drive/MyDrive/redes/gz2_hart16.csv", sep=',')

# abrindo tabela do kaggle
gx2 = pd.read_csv("/content/drive/MyDrive/redes/gz2_filename_mapping.csv", sep=',')

# agrupando através do inner join
gx = gx1.merge(gx2, how='inner')

# filtrando informações que iremos utilizar
colunas = ['dr7objid', 'gz2_class', 'asset_id']
gx = gx[colunas]

# extraindo caminhos de todas as imagens
img_end = glob(os.path.join('images','*.jpg'))

# gerarando dataframe com os caminhos
im = pd.DataFrame(img_end, columns=["images"])

# gerando caminho das imagens com o asset_id para comparação com imagens[images]
gx['images'] = gx['asset_id'].map(lambda num: "images/" + str(num) + ".jpg")

# agrupando os caminhos das imagens no dataset completo
gx = pd.merge(im, gx, how='inner', on='images')

# removendo imagens de estrelas
gx = gx[gx["gz2_class"] != "A"]

# criando uma nova coluna com a classificação binária de elipcas [1] e espirais [0]
gx['gz2_class'] = gx['gz2_class'].map(lambda rotulo: '0' if 'S' in rotulo else '1')

rotulos = gx['gz2_class'].drop_duplicates()
rotulos = sorted(rotulos)

gx.to_csv("tabelafinal.csv")
gx = pd.read_csv("tabelafinal.csv", sep=',')

# criando diretórios
for pasta in rotulos:
  criarDiretorio(diretorio, 'train', pasta)
  criarDiretorio(diretorio, 'test', pasta)
  criarDiretorio(diretorio, 'validation', pasta)

# separação de imagens em pastas
cont = 0
for r in rotulos:
    r = f"'{r}'"
    print(r)
    listaImagensRotuladas = geradorListaRotulos(r)
    random.shuffle(listaImagensRotuladas)

    for nome in listaImagensRotuladas:
        img = cv2.imread(nome)
        img = cv2.resize(img, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)

        if cont < len(listaImagensRotuladas) * 0.7:
            print(f'{cont} | {nome} | treino - tipo: {r}')
            cv2.imwrite(os.path.join(f'train/{r}', nome[7:]), img)
            cont += 1

        elif len(listaImagensRotuladas) * 0.7 < cont <= len(listaImagensRotuladas) * 0.9:
            print(f'{cont} | {nome} | teste - tipo: {r}')
            cv2.imwrite(os.path.join(f'test/{r}', nome[7:]), img)
            cont += 1

        else:
            print(f'{cont} | {nome} | validação - tipo: {r}')
            cv2.imwrite(os.path.join(f'validation/{r}', nome[7:]), img)
            cont += 1


directory_train = 'train'
directory_test = 'test'
directory_validation = 'validation'

train_ds = separarParaTTV(directory_train, 299)
test_ds = separarParaTTV(directory_test, 299)
val_ds = separarParaTTV(directory_validation, 299)


model = tf.keras.applications.InceptionV3(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1,
    classifier_activation="softmax",
)

root_logdir = os.path.join(os.curdir, "logs-fit")

def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.SGD(learning_rate=0.001), metrics=['accuracy'])
model.summary()

model.fit(train_ds, epochs=10, validation_data=val_ds, validation_freq=1, callbacks=[tensorboard_cb])
model.evaluate(test_ds)
