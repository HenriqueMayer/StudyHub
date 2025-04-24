# Dataset:  https://huggingface.co/datasets/ashraq/esc50
# numpy==2.1 (conda: project_audio)
from datasets import load_dataset
dataset = load_dataset('ashraq/esc50')

# Test and Analysis
dados = dataset['train']
dados[0] # Library: librosa

# Visualizar:
from matplotlib import pyplot as plt

idx_dados = 5
linha = dados[idx_dados]
plt.subplots(figsize=(30,4))
plt.plot(linha['audio']['array'])
plt.suptitle(linha['category'])

# Extrair para um arquivo: --------------------------------------------
from pathlib import Path

pasta_output = Path('Audio_py/audios_extraidos') / 'Objetos'

# exist_ok=True -> Se existir uma pasta com esse nome ele não vai retornar erro
# parents=True -> Ele vai criar na ordem que defini
pasta_output.mkdir(exist_ok=True, parents=True) # exist_ok=True

# ---------------------------------------------------------------------
# Separando quantos audios devem ser extraidos:
primeiras_linhas = dados.select(range(10)) 

import soundfile
for i, linha in enumerate(primeiras_linhas): # enumerate = retornar um indice junto
    objeto = linha['category']
    dados_som = linha['audio']['array']
    taxa_amostragem = linha['audio']['sampling_rate']
    caminho_output = pasta_output / f'{i}_{objeto}.wav'
    soundfile.write(file=caminho_output, data=dados_som, samplerate=taxa_amostragem)