# Dataset: https://huggingface.co/datasets/PolyAI/minds14
from datasets import load_dataset
ds_name = 'PolyAI/minds14'
ds_language = 'pt-PT'
ds = load_dataset(ds_name, name=ds_language, split='train[:10]', trust_remote_code=True)

# Model:
from transformers import pipeline
model_1 = 'openai/whisper-medium'
transcription = pipeline('automatic-speech-recognition', model=model_1)

# Checking the sampling_rate:
transcription.feature_extractor.sampling_rate # return: 16000
ds[0] # return: 8000

# O whisper converte automaticamente, precisa fornecer ao invés do array, o 'audio'.
transcription(ds[0]['audio'])

# //////////////////////////////////////
'''
Criando tarefas:
'''
model_1 = 'openai/whisper-medium'
transcription = pipeline(
    'automatic-speech-recognition', 
    model=model_1,
    generate_kwargs={'task': 'transcribe', 'language': 'portuguese'}
    ) # task: translate, ...