# Model: https://huggingface.co/facebook/mms-tts-por

import time
import IPython.display
from transformers import pipeline
model_1 =  'facebook/mms-tts-por'
reader = pipeline('text-to-speech', model=model_1)

# Texto:
text_1 = 'Olá, meu nome é Henrique.'

# Gerando áudio:
start = time.time()
speek = reader(text_1)
end = time.time()

f'{end-start:.2f} sec'

# Mostrar o áudio:
import IPython
IPython.display.Audio(data=speek['audio'], rate=speek['sampling_rate'])

# //////////////////////////////////////////////////////////////////////////////////////////

# Testando o model: suno/bark-small
model_2 =  'suno/bark-small'
reader_2 = pipeline('text-to-speech', model=model_2, forward_params={'max_new_tokens': 50})

# Texto:
text_1 = 'Olá, meu nome é Henrique. Sou o namorado da Alyne'

# Gerando áudio:
start = time.time()
speek_2 = reader_2(text_1)
end = time.time()

f'{end-start:.2f} sec'

# Mostrar o áudio:
import IPython
IPython.display.Audio(data=speek_2['audio'], rate=speek_2['sampling_rate'])

# //////////////////////////////////////////////////////////////////////////////////////////

# Utilizando o CUDA:
# Verificando se tem o CUDA
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'