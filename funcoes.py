import os

# funções
def criarDiretorio(diretorio: str, pasta: str, classificacao: str):
    # criando as pastas no diretório do projeto
    if os.path.isdir(pasta):
        os.mkdir(f'{diretorio}/{pasta}/{classificacao}')
    else:
        os.mkdir(f'{diretorio}/{pasta}')
        os.mkdir(f'{diretorio}/{pasta}/{classificacao}')

def geradorListaRotulos(rotulo: str):
    rotulo = gx.query(f'gz2_class == {rotulo}')
    imagensRotuladas = list(rotulo['images'])
    return imagensRotuladas


def separarParaTTV(diretorio: str, tamanhoImg: int):
  ds = tf.keras.utils.image_dataset_from_directory(
    diretorio,
    labels='inferred',
    label_mode='binary',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(tamanhoImg, tamanhoImg),
    shuffle=True,
    interpolation='bilinear', follow_links=False,
    crop_to_aspect_ratio=False
    )
  return ds