import numpy as np
from matplotlib import pyplot as plt
from math import floor
from PIL import Image


def carregarImagem(path):
    """
    Carrega a imagem em um objeto do tipo Image e retorna esse objeto.
    Parametro path: caminho do diretório onde se encontra a imagem concatenado com o nome da imagem e seu formato
    Retorno: objeto Image
    """

    image = Image.open(path)
    image.thumbnail((400, 400))
    return image


def salvarImagemJPEG(image, name):
    """
    Salva um objeto Image no disco.
    Parametro image: objeto Image a ser salvo
    Parametro name: nome sob o qual a imagem deve ser salva
    """

    image.save(name + ".jpg")

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

def calcularHistogramaImagem(image):
    """
    Calcula o histograma de uma imagem em tons de cinza.
    Parametro image: objeto Image cinza ou colorida
    Retorno: um array de inteiros
    """

    if(image.mode != "L"):
        image = converterImagemParaCinza(image)

    histogram = [0 for x in range(256)]

    image_np = np.asarray(image)
    
    for y in range(image_np.shape[0]):
        for x in range(image_np.shape[1]):
            histogram[image_np[y][x]]+=1
    
    return histogram

def ajustarBrilhoImagem(image, brightness):
    """
    Ajusta o brilho de uma imagem em tons de cinza ou com 3 canais (RGB).
    Parametro image: objeto Image cujo brilho deve ser ajustado
    Parametro brightness: valor do ajuste do brilho
    Retorno: objeto Imagem
    """ 

    if(image.mode == "L"):
        image_np = np.array(image)

        for y in range(image_np.shape[1]):
            for x in range(image_np.shape[0]):
                image_np[x][y] = truncar(image_np[x][y] + brightness)
                
    else:
        if(image.mode != "RGB"):
            image = converterImagemParaRGB(image)
        
        image_np = np.array(image)
            
        for y in range(image_np.shape[1]):
            for x in range(image_np.shape[0]):
                r,g,b = image_np[x][y]

                r = truncar(r + brightness)
                g = truncar(g + brightness)
                b = truncar(b + brightness)

                image_np[x][y] = (r,g,b)        
    
    return Image.fromarray(image_np)

def ajustarContrasteImagem(image, contrast):
    """
    Ajusta o contraste de uma imagem em tons de cinza ou com 3 canais (RGB).
    Parametro image: objeto Image cujo contraste deve ser ajustado
    Parametro contrast: valor do ajuste do contraste
    Retorno: objeto Image
    """ 

    correction_factor = (259 * (255 + contrast))/(255 * (259 - contrast))

    if(image.mode == "L"):
        image_np = np.array(image)

        for y in range(image_np.shape[1]):
            for x in range(image_np.shape[0]):
                image_np[x][y] = truncar(correction_factor * (image_np[x][y] - 128) + 128)
                
    else:
        if(image.mode != "RGB"):
            image = converterImagemParaRGB(image)
        
        image_np = np.array(image)
            
        for y in range(image_np.shape[1]):
            for x in range(image_np.shape[0]):
                r,g,b = image_np[x][y]
             
                r = truncar(correction_factor * (r - 128) + 128)
                g = truncar(correction_factor * (g - 128) + 128)
                b = truncar(correction_factor * (b - 128) + 128)

                image_np[x][y] = (r,g,b)        
    
    return Image.fromarray(image_np)

def calcularNegativoImagem(image):
    """
    Calcula o negativo de uma imagem em tons de cinza ou com 3 canais (RGB).
    Parametro image: objeto Image
    Retorno: objeto Imagem
    """ 

    image_np = np.array(image)

    if(image.mode == "L"):
        for y in range(image_np.shape[1]):
            for x in range(image_np.shape[0]):
                image_np[x][y] = 255 - image_np[x][y]

    else:
        if(image.mode != "RGB"):
            image = converterImagemParaRGB(image)
        
        for y in range(image_np.shape[1]):
            for x in range(image_np.shape[0]):
                r,g,b = image_np[x][y]

                r = 255 - r
                g = 255 - g
                b = 255 - b

                image_np[x][y] = (r,g,b)
    
    return Image.fromarray(image_np)

def calcularHistogramaAcumuladoImagem(image):
    """
    Calcula o histograma acumulado de uma imagem.
    Parametro image: objeto Image da imagem base
    Retorno: array de inteiros
    """

    image_np = np.array(image)

    hist = []
    acumulated_hist = [0 for x in range(256)]
    alfa = 255.0 / (image_np.shape[0] * image_np.shape[1])

    if(image.mode != "L"):
        image = converterImagemParaCinza(image)

    hist = calcularHistogramaImagem(image)
    acumulated_hist[0] = alfa * hist[0]

    for i in range(1,256):
        acumulated_hist[i] = acumulated_hist[i - 1] + alfa * hist[i]
    
    return acumulated_hist

def equalizarHistogramaImagem(image):
    """
    Remolda os valores dos pixels de uma imagem com base no seu histograma acumulado.
    Parametro image: objeto Image da imagem base
    Retorno: objeto Image
    """ 

    image_np = np.array(image)

    # cria um array não inicializado do mesmo formato da imagem
    new_image_list = np.empty_like(image_np)
    acumulated_hist = calcularHistogramaAcumuladoImagem(image)

    if(image.mode == "L"):
        for y in range(image_np.shape[0]):
            for x in range(image_np.shape[1]):
                new_image_list[y][x] = acumulated_hist[image_np[y][x]]
    
    else:
        for y in range(image_np.shape[0]):
            for x in range(image_np.shape[1]):
                for c in range(3):
                    new_image_list[y][x][c] = acumulated_hist[image_np[y][x][c]]
    
    return Image.fromarray(np.asarray(new_image_list))

def matchHistogramaImagem(img_source, img_target):
    """
    Mapeia a imagem fonte para os valores dos pixels da imagem target.
    Parametro img_source: objeto Image a receber o valor dos pixels da imagem target
    Parametro img_target: objeto Image que fornece os novos valores de pixel
    Retorno: objeto Image
    """ 

    if(img_source.mode != "L"):
        img_source = converterImagemParaCinza(img_source)
        img_target = converterImagemParaCinza(img_target)

    hist_source_ac = calcularHistogramaAcumuladoImagem(img_source)
    hist_target_ac = calcularHistogramaAcumuladoImagem(img_target)
    hist_matching = [0 for x in range(256)]

    img_source_np = np.array(img_source)
    image_hm_np = np.empty_like(img_source_np)

    for shade_index in range(256):
        hist_matching[shade_index] = encontrarTomMaisProximoDe(shade_index, hist_source_ac, hist_target_ac)
    
    for y in range(img_source_np.shape[0]):
        for x in range(img_source_np.shape[1]):
            image_hm_np[y][x] =  hist_matching[img_source_np[y][x]]
    
    return Image.fromarray(image_hm_np)

def rotacionarImagem90GrausAntiHorario(image):
    """
    Rotaciona a imagem em 90 graus no sentido antihorário.
    Parametro image: objeto Image a ser rotacionado
    Retorno: objeto Image rotacionada
    """
    image_np = np.array(image)
    image_90 = []

    for x in range(image_np.shape[1] - 1, -1, -1):
        new_row = []

        for y in range(image_np.shape[0]):
            new_row.append(image_np[y][x])
        
        image_90.append(new_row)
    
    return Image.fromarray(np.array(image_90))

def rotacionarImagem90GrausHorario(image):
    """
    Rotaciona a imagem em 90 graus no sentido horário.
    Parametro image: objeto Image a ser rotacionado
    Retorno: objeto Image rotacionada
    """ 
    image_np = np.array(image)
    image_90 = []

    for x in range(image_np.shape[1]):
        new_row = []

        for y in range(image_np.shape[0] - 1, -1, -1):
            new_row.append(image_np[y][x])
        
        image_90.append(new_row)
            
    return Image.fromarray(np.array(image_90))

def aplicarConvolucaoKernel_3x3(image, kernel):
    """
    Faz a convolução de acordo com o kernel 3x3 fornecido.
    Parametro image: objeto Image
    Parametro kernel: uma matriz de tamanho 3x3
    Retorno: objeto Image em tons de cinza e com o filtro aplicado.
    """ 

    if(image.mode != "L"):
        image = converterImagemParaCinza(image)

    image_np = np.array(image)
    image_out = np.empty_like(image_np)

    # M e m são colunas
    m = len(kernel)
    m2 = floor(m/2)
    M = image_np.shape[1]

    # N e n são linhas 
    n = len(kernel[0])
    n2 = floor(n/2)
    N = image_np.shape[0]

    for y in range(n2, (N - n2)):
        for x in range(m2, (M - m2)):
            summation = 0.0
            
            for k in range(-n2, n2 + 1):
                for j in range(-m2, m2 + 1):
                    summation += (kernel[m2 + j][n2 + k] * image_np[y - k][x - j])
            
            if(confereSeKernelPassaAlto(kernel)):
                image_out[y][x] = truncar(summation)
            else:
                image_out[y][x] = truncar(summation + 127)

    return Image.fromarray(image_out)

def aumentarImagem(image):
    """
    Aumenta o tamanho da imagem com intepolação linear 2x2.
    Parametro image: objeto Image a ser aumentado
    Retorno: objeto Image
    """
    if(image.mode != "L" or image.mode != "RGB"):
        image = converterImagemParaRGB(image)

    image_np = np.array(image)
    image_aux = []
    image_out = []

    rows = image_np.shape[0]
    columns = image_np.shape[1]

    # interpolação nas linhas (inserindo colunas)
    for y in range(rows):
        new_line = []

        for x in range(columns):
            new_line.append(image_np[y][x])

            # confere se não é a última coluna
            if(x != columns-1):

                # insere pixel entre colunas
                if(image.mode == "L"):
                    new_pixel = ((image_np[y][x] + image_np[y][x+1])/2)
                    new_line.append(new_pixel)
                else:
                    new_pixel = calculaMediaPixelRGB(image_np[y][x], image_np[y][x+1])
                    new_line.append(new_pixel)

        image_aux.append(new_line)
    
    rows = len(image_aux)
    columns = len(image_aux[0])

    # interpolação nas colunas (inserindo linhas)
    for y in range(rows):
        line = []
        new_line = []

        for x in range(columns):
            line.append(image_aux[y][x])

            # confere se não é a última linha
            if(y != rows-1):
                if(image.mode == "L"):
                    new_pixel = ((image_aux[y][x] + image_aux[y][x+1])/2)
                    new_line.append(new_pixel)
                else:
                    new_pixel = calculaMediaPixelRGB(image_aux[y][x], image_aux[y+1][x])
                    new_line.append(new_pixel)

        image_out.append(line)

        if(y != rows-1):
            image_out.append(new_line)
    
    return Image.fromarray(np.array(image_out))

def reduzirImagem(image, Sx, Sy):
    """
    Reduz o tamanho da imagem de acordo com os fatores de redução.
    Parametro image: objeto Image a ser reduzida
    Parametro Sx: fator de redução x
    Parametro Sy: fator de redução y
    Retorno: objeto Image
    """

    image_np = np.array(image)
    width  = image_np.shape[1]
    height = image_np.shape[0]

    new_image = []

    # move o retangulo sobre a imagem
    for y in range(0, height, Sy):
        pixel_row = []
        for x in range(0, width, Sx):

            # encontra os limites do retangulo
            x_sup = x
            x_inf = width if x+Sx > width else x+Sx
            y_sup = y
            y_inf = height if y+Sy > height else y+Sy

            rn = 0
            gn = 0
            bn = 0

            # dentro de cada retangulo
            img_slice = image_np[y_sup:y_inf, x_sup:x_inf]

            for row in img_slice:

                # soma os canais individualmente
                for r,g,b in row:
                    rn += r
                    gn += g
                    bn += b

            # divide cada canal pela quantidade de pixels
            pixel_num = img_slice.shape[0]*img_slice.shape[1]

            # adiciona esse pixel a linha de pixels nova
            pixel_row.append((rn/pixel_num, gn/pixel_num, bn/pixel_num))
			
        # coloca a linha de pixels na imagem
        new_image.append(pixel_row)

    return Image.fromarray(np.array(new_image, dtype=np.uint8))

def converterImagemParaRGB(image):
    """
    Converte a imagem recebida para o mode RGB.
    Parametro image: objeto Image a ser convertido em RGB
    Retorno: objeto Image em RGB
    """ 

    return image.convert("RGB")

def encontrarTonsIntensidade(image):
    """
    Encontra qual é o menor e o maior tom de intensidade de uma imagem.
    Parametro image: objeto imagem
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

def truncar(pixel):
    """
    Garante que o valor de um pixel esteja entre o intervalo (0,255).
    Parametro pixel: inteiro
    Retorno: um inteiro entre (0,255) 
    """

    if(pixel < 0):
        return 0
    elif(pixel > 255):
        return 255
    else:
        return pixel

def encontrarTomMaisProximoDe(shade_index, hist_source_ac, hist_target_ac):
    """
    Encontra no histograma acumulado target o tom mais próximo do tom no histograma acumulado source cujo índice é shade_index.
    Parametro shade_index: índice do tom no histograma fonte.
    Parametro hist_source_ac: histograma fonte acumulado
    Parametro hist_target_ac: histograma target acumulado
    Retorno: inteiro, que é o tom mais próximo
    """
    shade_src = []

    for x in range(256):
        shade_src.append(hist_source_ac[shade_index])

    index_shade_trgt = (np.abs(np.subtract(hist_target_ac, shade_src))).argmin()
    return hist_target_ac[index_shade_trgt]

def confereSeKernelPassaAlto(kernel):
    """
    Confere se o kernel fornecido é um kernel para filtro passa-alta.
    Parametro kernel: uma matriz 3x3
    Retorno: Boolean
    """
    k_gaussian = [[0.0625, 0.125, 0.0625],
                   [0.125, 0.25, 0.125],
                   [0.0625, 0.125, 0.0625]]
    
    k_laplacian = [[0, -1, 0],
                   [-1, 4, -1],
                   [0, -1, 0]]
    
    k_generic = [[-1, -1, -1],
                 [-1, 8, -1],
                 [-1, -1, -1]]
    
    if(kernel == k_gaussian or kernel == k_laplacian or kernel == k_generic):
        return True
    else:
        return False

def calculaMediaPixelRGB(rgb_pixel1, rgb_pixel2):
    """
    Calcula a média das bandas de um pixel RGB.
    Parametro rgb_pixel1: tupla (r,g,b)
    Parametro rgb_pixel2: tupla (r,g,b)
    Retorno: tupla (r,g,b)
    """
    r1, g1, b1 = rgb_pixel1
    r2, g2, b2 = rgb_pixel2

    rm = np.uint8((int(r1) + int(r2))/2)
    gm = np.uint8((int(g1) + int(g2))/2)
    bm = np.uint8((int(b1) + int(b2))/2)

    return (rm, gm, bm)



image_original = Image.open("test_images/Gramado_22k.jpg")

# HISTOGRAMA
# image_original.show()
# plt.plot(range(0,256), calcularHistogramaImagem(image_original))
# plt.show()

# AJUSTAR BRILHO
# ajustarBrilhoImagem(image_original, 120).show()
# ajustarBrilhoImagem(image_original, -70).show()

# AJUSTAR CONTRASTE
# ajustarContrasteImagem(image_original, 150).show()
# ajustarContrasteImagem(image_original, -150).show()

# CALCULAR NEGATIVO
# calcularNegativoImagem(image_original).show()

# CALCULAR HISTOGRAMA ACUMULADO DE IMAGEM CINZA
# image_gray = converterImagemParaCinza(image_original)
# image_gray.show()
# plt.plot(range(0,256), calcularHistogramaAcumuladoImagem(image_gray))
# plt.show()

# CALCULAR HISTOGRAMA ACUMULADO DE IMAGEM COLORIDA
# image_original.show()
# plt.plot(range(0,256), calcularHistogramaAcumuladoImagem(image_original))
# plt.show()

# EQUALIZAR HISTOGRAMA
# image_original.show()
# equalizarHistogramaImagem(image_original).show()

# MATCH HISTOGRAMA
# image_gray = converterImagemParaCinza(image_original)
# image_gray.show()
# image_aux = Image.open("test_images/Underwater_53k.jpg")
# image_aux_gray = converterImagemParaCinza(image_aux)
# matchHistogramaImagem(image_gray, image_aux_gray).show()

# ROTACIONAR IMAGEM SENTIDO HORARIO E ANTIHORARIO
# rotacionarImagem90GrausHorario(image_original).show()
# rotacionarImagem90GrausAntiHorario(image_original).show()

## CONVOLUCAO
# APLICAR FILTRO GAUSSIANO
# k_gaussian = [[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]]
# aplicarConvolucaoKernel_3x3(image_original, k_gaussian).show()

# APLICAR FILTRO LAPLACIANO
# k_laplacian = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
# aplicarConvolucaoKernel_3x3(image_original, k_laplacian).show()

# APLICAR FILTRO GENÉRICO
# k_generic = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
# aplicarConvolucaoKernel_3x3(image_original, k_generic).show()

# APLICAR FILTRO PREWITT HORIZONTAL
# k_prewwit_h = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
# aplicarConvolucaoKernel_3x3(image_original, k_prewwit_h).show()

# APLICAR FILTRO PREWITT VERTICAL
# k_prewwit_v = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
# aplicarConvolucaoKernel_3x3(image_original, k_prewwit_v).show()

# APLICAR SOBEL HORIZONTAL
# k_sobel_h = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
# aplicarConvolucaoKernel_3x3(image_original, k_sobel_h).show()

# APLICAR SOBEL VERTICAL
# k_sobel_v = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
# aplicarConvolucaoKernel_3x3(image_original, k_sobel_v).show()

# AUMENTAR IMAGEM
# image_original.show()
# aumentarImagem(image_original).show()

# REDUZIR IMAGEM
# image_original.show()
# reduzirImagem(image_original, 3, 3).show()