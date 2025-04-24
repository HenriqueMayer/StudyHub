# Import the dataset:
from datasets import load_dataset
ds_cat_dog = load_dataset('Bingsu/Cat_and_Dog', split='test')

# Visualizar:
imagens = ds_cat_dog.shuffle()[:10]
for image in imagens['image']:
    display(image)

def exibir_imagem(imagem):
    '''
    Exibir as imagens no formato 250x250
    '''
    imagem = imagem.copy() # Não alterar a original
    imagem.thumbnail((250,250))
    display(imagem)


# Model: https://huggingface.co/akahana/vit-base-cats-vs-dogs
from transformers import pipeline
modelo = 'akahana/vit-base-cats-vs-dogs'
classificador = pipeline('image-classification', model=modelo)

# aplicando no modelo:
for imagem in imagens['image']:
    prediction = classificador(imagem)
    print(prediction)
    exibir_imagem(imagem)

# Testando com imagens pessoais:
from pathlib import Path
from PIL import Image

imagens_teste = [
    Image.open(arquivo) 
    for arquivo in Path('Image_Cat_Dog').iterdir()
    ]

for imagem in imagens_teste:
    prediction = classificador(imagem)
    # prob_cat = prediction[0]['score']*100
    # prob_dog = prediction[0]['score']*100
    # print(f'A pobabilidade de ser:\nGato -> {prob_cat:.2f}%\nCachorro -> {prob_dog:.2f}%')
    print(prediction)
    exibir_imagem(imagem)