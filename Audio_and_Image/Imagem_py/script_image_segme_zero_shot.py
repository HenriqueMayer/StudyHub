# Model: https://huggingface.co/CIDAS/clipseg-rd64-refined
from transformers import pipeline
model_name = 'CIDAS/clipseg-rd64-refined'
segmenter = pipeline('image-segmentation', model=model_name)

# image_Setup:
from pathlib import Path
from PIL import Image

caminho = 'Image_Cities/image_3.jpg'
image = Image.open(caminho)

'''
Atenção: modelo não está pronto para ser utilizado por pipeline.
Solução: ir na documentation. 

'''
from transformers import CLIPProcessor, CLIPSegForImageSegmentation
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPSegForImageSegmentation.from_pretrained(model_name)

# O que o modelo deve procurar:
prompts = [
    'street',
    'cars',
    'trafic light',
]

inputs = processor(
    text=prompts, 
    images=[image]*len(prompts),
    padding=True,
    return_tensors='pt'
    )

# Machine Learning:
import torch
with torch.no_grad():
    outputs = model(**inputs) # Inputs entrando no modelo

predictions = outputs.logits.unsqueeze(1)

segmentation = []
for i, label in enumerate(prompts):
    d = {'label': label, 'mask': torch.sigmoid(predictions[i][0]).numpy()}
    segmentation.append(d)

segmentation

# Visualization:
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=1, ncols=len(prompts)+1, figsize=(16,4)) # '+1' para ter a imagem original

axes[0].imshow(image)
for i, segmento in enumerate(segmentation):
    axes[i+1].imshow(segmento['mask'], cmap='viridis')
    axes[i+1].set_title(segmento['label'])

fig