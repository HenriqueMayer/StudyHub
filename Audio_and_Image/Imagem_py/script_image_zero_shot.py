from transformers import pipeline

# Model:
model_1 = 'openai/clip-vit-large-patch14'
classifier = pipeline('zero-shot-image-classification', model=model_1)

# func para exibir imagem:
def show_images(imagem):
    imagem = imagem.copy()
    imagem.thumbnail((250,250))
    display(imagem)

# navegar pelas pastas:
from pathlib import Path
from PIL import Image

img_animals = [Image.open(arquivo)
               for arquivo in Path('Image_Cat_Dog').iterdir()]

# candidate_labels
labels = [
    'dog',
    'cat',
    'black bear',
    'wolf',
    'tiger',
    'bird',
]

for image in img_animals:
    prediction = classifier(image, candidate_labels=labels)
    print('----Predictions----')
    for pred in prediction:
        label = pred['label']
        score = pred['score']
        score_aj = f'{100*score:.2f}%'
        print(f'{label}: {score_aj}')
    show_images(image)