import os

def criarDiretorio(diretorio: str, pasta: str, classificacao: str):
    # criando as pastas no diretório do projeto
    os.mkdir(f'{diretorio}/{pasta}')
    os.mkdir(f'{diretorio}/{pasta}/{classificacao}')
