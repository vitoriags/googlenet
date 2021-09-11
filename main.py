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
# diretorio = os.getcwd()
# print(diretorio)

# executado apenas uma vez
# criarDiretorio(diretorio, 'train', '0')
# criarDiretorio(diretorio, 'train', '1')
# criarDiretorio(diretorio, 'validation', '0')
# criarDiretorio(diretorio, 'validation', '1')

# os.makedirs('../train299/0')
# os.makedirs('../train299/1')
# os.makedirs('../test299/0')
# os.makedirs('../test299/1')
# os.makedirs('../validation299/0')
# os.makedirs('../validation299/1')


# # abrindo tabela de rótulos
# gx1 = pd.read_csv("gz2_hart16.csv", sep=',')
#
# # abrindo tabela do kaggle
# gx2 = pd.read_csv("gz2_filename_mapping.csv", sep=',')
#
# # agrupando através do inner join
# gx = gx1.merge(gx2, how='inner')
#
# # filtrando informações que iremos utilizar
# colunas = ['dr7objid', 'gz2_class', 'asset_id']
# gx = gx[colunas]
#
# # extraindo caminhos de todas as imagens
# img_end = glob(os.path.join('../images','*.jpg'))
#
# # gerarando dataframe com os caminhos
# im = pd.DataFrame(img_end, columns=["images"])
# print(im)
#
# # corrigindo os caminhos
# im["images"] = im["images"].str.replace('\\', '/')
#
# # gerando caminho das imagens com o asset_id para comparação com imagens[images]
# gx['images'] = gx['asset_id'].map(lambda num: "../images/" + str(num) + ".jpg")
#
# # agrupando os caminhos das imagens no dataset completo
# gx = pd.merge(im, gx, how='inner', on='images')
#
# # removendo imagens de estrelas
# gx = gx[gx["gz2_class"] != "A"]
#
# # criando uma nova coluna com a classificação binária de elipcas [1] e espirais [0]
# gx['newclass'] = gx['gz2_class'].map(lambda num: 1 if 'E' in num else 0)
#
# gx.to_csv("tabelafinal.csv")
# gx = pd.read_csv("tabelafinal.csv", sep=',')
#
# # carregando lista do caminho das imagens
# galaxias = list(gx['images'])
#
# # selecionando as linhas onde 'newclass' é igual a 0 e guardando as informações de 'images' em uma lista
# esp = gx.query('newclass == 0')
# spirals = list(esp['images'])
#
# # randomizando vetor de caminhos das imagens
# random.shuffle(galaxias)
#
# # leitura/resize/gravação em diretórios separados
# cont = 0
# for nome in galaxias:
#     img = cv2.imread(nome)
#     img = cv2.resize(img, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)
#
#     if cont <= len(galaxias) * 0.7:
#         if nome in spirals:
#             print(f'{cont} | {nome} | treino - tipo: 0')
#             cv2.imwrite(os.path.join('../train299/0', nome[10:]), img)
#             cont += 1
#         else:
#             print(f'{cont} | {nome} | treino - tipo: 1')
#             cv2.imwrite(os.path.join('../train299/1', nome[10:]), img)
#             cont += 1
#
#     elif len(galaxias) * 0.7 < cont <= len(galaxias) * 0.9:
#         if nome in spirals:
#             print(f'{cont} | {nome} | teste - tipo: 0')
#             cv2.imwrite(os.path.join('../test299/0', nome[10:]), img)
#             cont += 1
#         else:
#             print(f'{cont} | {nome} | teste - tipo: 1')
#             cv2.imwrite(os.path.join('../test299/1', nome[10:]), img)
#             cont += 1
#
#     else:
#         if nome in spirals:
#             print(f'{cont} | {nome} | validação - tipo: 0')
#             cv2.imwrite(os.path.join('../validation299/0', nome[10:]), img)
#             cont += 1
#         else:
#             print(f'{cont} | {nome} | validação - tipo: 1')
#             cv2.imwrite(os.path.join('../validation299/1', nome[10:]), img)
#             cont += 1

directory_train = '../train299'
directory_test = '../test299'
directory_validation = '../validation299'


train_ds = tf.keras.utils.image_dataset_from_directory(
    directory_train,
    labels='inferred',
    label_mode='binary',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(299, 299),
    shuffle=True,
    interpolation='bilinear', follow_links=False,
    crop_to_aspect_ratio=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    directory_test,
    labels='inferred',
    label_mode='binary',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(299, 299),
    shuffle=True,
    interpolation='bilinear', follow_links=False,
    crop_to_aspect_ratio=False
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    directory_validation,
    labels='inferred',
    label_mode='binary',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(299, 299),
    shuffle=True,
    interpolation='bilinear', follow_links=False,
    crop_to_aspect_ratio=False
)

model = tf.keras.applications.InceptionV3(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1,
    classifier_activation="softmax",
)
# keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None)



root_logdir = os.path.join(os.curdir, "logs\\fit\\")

def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.SGD(learning_rate=0.001), metrics=['accuracy'])

model.summary()

model.fit(train_ds, epochs=20, validation_data=val_ds, validation_freq=1, callbacks=[tensorboard_cb])

model.evaluate(test_ds)
