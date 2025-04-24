# Model: https://huggingface.co/nvidia/segformer-b1-finetuned-cityscapes-1024-1024
from transformers import pipeline
model_1 = 'nvidia/segformer-b1-finetuned-cityscapes-1024-1024'
segmenter = pipeline('image-segmentation', model=model_1)

# Setup:
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

image_1 = Image.open('Image_Cities/image_2.jpg')
segmentation = segmenter(image_1)

# How the 'mask' works in image segmentation:
road_segmentation = segmentation[0] 
road_segmentation['mask']

# Compare/Analysis:
image_1_data = np.array(image_1)
road_segmentation_mask_data = np.array(road_segmentation['mask'])
image_1_data.shape
road_segmentation_mask_data.shape


# Transferir os valores da imagem original para a mascarada:
# A imagem 'mask' não tem 3 dimensões, então eu adiciono uma com o 'np.newaxis'
mask_image = image_1_data & road_segmentation_mask_data[:,:, np.newaxis]
plt.imshow(mask_image)

mask_image_inverse_data = np.bitwise_not(road_segmentation_mask_data)
mask_image_inverse= image_1_data & mask_image_inverse_data[:,:, np.newaxis]
plt.imshow(mask_image_inverse)


# Adicionar cores/transparência/fundo_verde:
color_green = np.array([0, 255, 0])
image_1_green = image_1_data.copy()
image_1_green = np.where(road_segmentation_mask_data[:,:,np.newaxis] == 255, color_green, image_1_green)
plt.imshow(image_1_green)


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////
from pathlib import Path

import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.patches import Patch
import numpy as np
from transformers import pipeline

# Carregar modelo
modelo = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
segmentador = pipeline("image-segmentation", model=modelo)

def adicionar_mascara(dados_imagem, dados_mascara, cor):
    imagem_mascara = dados_imagem.copy()
    imagem_mascara = np.where(
        dados_mascara[:, :, np.newaxis] == 255, 
        cor, 
        imagem_mascara,
    ).astype(dados_imagem.dtype)
    return cv2.addWeighted(dados_imagem, 0.5, imagem_mascara, 0.5, 0)


def plotar_segmentos(imagem, segmentacao, nome_colormap):
    # Pegar cores de output
    cmap = colormaps.get_cmap(nome_colormap)
    cores = [
        (np.array(cmap(x)[:3]) * 255).astype(int) 
        for x in np.linspace(0, 1, len(segmentacao))
    ]
    
    # Apresentar output em imagem de saída
    imagem_final = np.array(imagem).copy()
    legendas = []
    for segmento, cor in zip(segmentacao, cores):
        dados_mascara = np.array(segmento['mask'])
        label_mascara = segmento['label']
        imagem_final = adicionar_mascara(imagem_final, dados_mascara, cor)
        legendas.append(Patch(facecolor=cor/255, edgecolor='black', label=label_mascara))
    
    fig, ax = plt.subplots(figsize=(16, 16))
    plt.legend(handles=legendas)
    
    ax.imshow(imagem_final)

# Carregar dados de entrada
nome_colormap = 'hot'
imagem = Image.open(Path('Image_Cities/image_2.jpg'))

# Rodar modelo
segmentacao = segmentador(imagem)

# Mostrar resultados
plotar_segmentos(imagem=imagem, segmentacao=segmentacao, nome_colormap=nome_colormap)