# Model: facebook/detr-resnet-50
from transformers import pipeline
model_name = 'facebook/detr-resnet-50'

detector = pipeline('object-detection', model=model_name)

# Image-Setup
from pathlib import Path
from PIL import Image
image = Image.open('Image_Cities/image_4.jpg')

# Detector:
detector_1 = detector(image)

from matplotlib import patches
import matplotlib.pyplot as plt
fig, axes = plt.subplots(figsize=(16,16))
axes.imshow(image)

det = detector_1[0]
box = det['box']
origem = (box['xmin'], box['ymin'])
largura = box['xmax'] - box['xmin']
altura = box['ymax'] - box['ymin']

rect = patches.Rectangle(
    origem, 
    largura, 
    altura, 
    linewidth=3,
    edgecolor='red',
    facecolor='none'
    )
axes.add_patch(rect)
texto = f'{det["label"]} {100*det["score"]:.2f}%'
axes.text(box['xmin'], box['ymin'] - 2, texto, bbox={'facecolor': 'red', 'alpha':0.8})
fig

# ///////////////////////////////////////////////////////////////////////////////////////
# Função para detectar todos os objetos:
def plotar_detec(img, detc):
    fig, axes = plt.subplots(figsize=(16,16))
    axes.imshow(img)

    for detection in detc:
        box = detection['box']
        origem = (box['xmin'], box['ymin'])
        largura = box['xmax'] - box['xmin']
        altura = box['ymax'] - box['ymin']

        rect = patches.Rectangle(origem,largura,altura,linewidth=3,edgecolor='red',facecolor='none')
        axes.add_patch(rect)

        text = f'{detection["label"]} {100*detection["score"]:.2f}%'
        axes.text(box['xmin'], box['ymin'] - 2, texto, bbox={'facecolor': 'red', 'alpha':0.8})

plotar_detec(img=image, detc=detector_1)