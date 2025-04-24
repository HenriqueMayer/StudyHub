# Dataset: google/fleurs
from datasets import load_dataset
nome_dataset = 'google/fleurs'
lingua_dataset = 'pt_br'

# streaming-True -> Para não baixar os dados do dataset, pois podem ser muito pesados
dataset_fleurs = load_dataset(nome_dataset, name=lingua_dataset, split='train', streaming=True, trust_remote_code=True)

# ---------------------
primeiras_linhas = dataset_fleurs.take(5) # Se não iterar ele não vai carregar:
for linhas in primeiras_linhas:
    print(linhas)

# Carregar o modelo:
from transformers import pipeline
import tensorflow
import torch
modelo = 'sanchit-gandhi/whisper-medium-fleurs-lang-id'
classificador_audio = pipeline('audio-classification', model=modelo)


# Infos:
classificador_audio.feature_extractor.sampling_rate # sampling_rate igual ao que os dados possuem
# Como verficar? no 'for' acima retorna informações sobre o audio, uma delas é: 'sampling_rate': 16000

# Para trabalhar com os dados preciso salvar eles em uma lista, tendo em vista que estão em streaming
primeiras_linhas = list(primeiras_linhas)
primeira_linha = primeiras_linhas[0]

# primeira_linha:
'''
{'id': 114,
 'num_samples': 312960,
 'path': None,
 'audio': {'path': 'train/10009971053374752024.wav',
  'array': array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, ...,
         9.54866409e-05, 1.48892403e-04, 1.79767609e-04]),
  'sampling_rate': 16000},
 'transcription': 'o governo anterior considerado conservador da austrália se negou a ratificar kyoto 
 afirmando que seria ruim para a economia por sua forte dependência das exportações de carvão ao mesmo tempo 
 que alguns países como índia e china não tinham limitações por metas de emissões',
 'raw_transcription': 'O governo anterior, considerado conservador, da Austrália se negou a ratificar Kyoto, 
 afirmando que seria ruim para a economia por sua forte dependência das exportações de carvão, ao mesmo tempo que alguns 
 países como Índia e China não tinham limitações por metas de emissões.',
 'gender': 0,
 'lang_id': 76,
 'language': 'Portuguese',
 'lang_group_id': 0}
'''

# Passando o array do audio para o classificador:
predic = classificador_audio(primeira_linha['audio']['array'])

# O modelo tem 99,999...% de certeza que é português o idioma do áudio
'''
[{'score': 0.9999843835830688, 'label': 'Portuguese'},
 {'score': 1.8095358882419532e-06, 'label': 'Kabuverdianu'},
 {'score': 1.3603338402390364e-06, 'label': 'Persian'},
 {'score': 6.062232387193944e-07, 'label': 'Urdu'},
 {'score': 5.789391366306518e-07, 'label': 'Northern-Sotho'}]
'''


# Extra: import sounddevice
# Realizar gravações diretas
'''
import sounddevice as sd

duracao=10 # 10 sec
taxa_amostragem=16000
tamanho_vetor=int(duracao*taxa_amostragem)

gravacao = sd.rec(tamanho_vetor, sampling_rate=taxa_amostragem, channels=1) # channels -> 1 == mono

gravacao retorna um array
-> deixar na horizontal: gravacao.reval()
'''