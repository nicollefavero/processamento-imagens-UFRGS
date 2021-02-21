import numpy as np
from math import floor
from PIL import Image


def carregarImagem(path):
    """
    Carrega a imagem em um objeto do tipo Image e retorna esse objeto.
    Parametro path: caminho do diretório onde se encontra a imagem concatenado com o nome da imagem e seu formato
    Retorno: objeto Image
    """

    return Image.open(path)


def salvarImagemJPEG(image, name):
    """
    Salva um objeto Image no disco.
    Parametro image: objeto Image a ser salvo
    Parametro name: nome sob o qual a imagem deve ser salva
    """

    image.save(name + ".jpeg")

def espelharImagemHorizontal(image):
    """
    Espelha horizontalmente a imagem passada (de qualquer mode) como parâmetro.
    Parametro image: imagem a ser espelhada
    Retorno: objeto Image
    """

    if(image.mode != "RGB"):
        image = converterImagemParaRGB(image)

    image_np = np.asarray(image)
    mirrored_image_np = []

    for line in image_np:
        mirrored_image_np.append(line[::-1]) 
        
    return Image.fromarray(np.array(mirrored_image_np))

def espelharImagemVertical(image):
    """
    Espelha verticalmente a imagem passada (de qualquer mode) como parâmetro.
    Parametro image: imagem a ser espelhada
    Retorno: objeto Image
    """

    if(image.mode != "RGB"):
        image = converterImagemParaRGB(image)

    image_np = np.asarray(image)
    return Image.fromarray(image_np[::-1])

def converterImagemParaCinza(image):
    """
    Converte a imagem original (de qualquer mode) para tons de cinza (luminância) conforme a equação L = 0.299*R + 0.587*G + 0.114*B.
    Parametro image: imagem a ser convertida em tons de cinza
    Retorno: objeto Image (mode L)
    """

    if(image.mode != "RGB"):
        image = converterImagemParaRGB(image)

    image_np = np.asarray(image)
    gray_image = []

    for line in image_np:

        pixels_line = []

        for r, g, b in line:
            L = 0.299 * r + 0.587 * g + 0.114 * b
            pixels_line.append(np.uint8(L))
        
        gray_image.append(pixels_line)

    return Image.fromarray(np.array(gray_image))

def quantizarImagemCinza(image, shades_num):
    """
    Reduz a quantidade de tons de uma imagem.
    Parametro image: imagem original a ser quantizada
    Parametro shades_num: nova quantidade de tons da imagem
    Retorno: objeto imagem (quantizada)
    """

    if(image.mode != "L"):
        image = converterImagemParaCinza(image)
    
    shade_min, shade_max = encontrarTonsIntensidade(image)
    intensity = (shade_max - shade_min + 1)

    if shades_num < intensity:
        bin_size = intensity/shades_num

        def calcularBinPontoMedio(index):
            return np.uint8((shade_min + (index*bin_size) + shade_min + ((index + 1)*bin_size))/2)

        bin_intervals_pm = [calcularBinPontoMedio(i) for i in range(shades_num)]

        image_np = np.asarray(image)
        quantized_image = []

        for line in image_np:

            pixels_line = []

            for pixel in line:
                index = floor((pixel - shade_min)/bin_size)
                pixels_line.append(bin_intervals_pm[index])
            
            quantized_image.append(pixels_line)
        
        return Image.fromarray(np.array(quantized_image))
    
    else:
        return image


def converterImagemParaRGB(image):
    """
    Converte a imagem recebida para o mode RGB.
    Parametro: imagem a ser convertida em RGB
    Retorno: imagem em RGB
    """ 

    return image.convert("RGB")

def encontrarTonsIntensidade(image):
    """
    Encontra qual é o menor e o maior tom de intensidade de uma imagem.
    Parametro: objeto imagem
    Retorno: um 2-tupla (m, M), sendo m o menor tom de intensidade e M o maior tom de intensidade 
    """
    
    image_np = np.asarray(image)

    shade_min = shade_max = image_np[0][0]

    for line in image_np:
        for pixel in line:
            if pixel < shade_min:
                shade_min = pixel
            elif pixel > shade_max:
                shade_max = pixel
    
    return (shade_min, shade_max)

            
if __name__ == "__main__":
    
    path = "test_images/Underwater_53k.jpg"

    image = carregarImagem(path)

    # salvarImagemJPEG(espelharImagemHorizontal(image), "horizontal-mirroring")
    # salvarImagemJPEG(espelharImagemVertical(image), "vertical-mirroring")
    # salvarImagemJPEG(converterImagemParaCinza(image), "gray-scale")
    # salvarImagemJPEG(quantizarImagemCinza(image, 4), "quantized-image")q