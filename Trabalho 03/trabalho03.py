import numpy as np
from cv2 import cv2

# Filtro Gaussiano
def aplicarBorramentoGaussiano(frame, kernel_size):
    # se o tamanho do kernel for par, pula para o próximo número ímpar
    kernel_size = kernel_size + (kernel_size%2 == 0)

    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

# Filtro Canny
def detectarArestas(frame):
    frame = cv2.Canny(frame, 50, 100)
    frame = np.expand_dims(frame, axis=-1)
    return np.concatenate((frame, frame, frame), axis=2)

# Filtro Sobel
def obterEstimativaGradiente(frame):
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    src = cv2.GaussianBlur(frame, (3,3), 0)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.convertScaleAbs(cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT))
    grad_y = cv2.convertScaleAbs(cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT))

    grad_frame = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
    grad_frame = np.expand_dims(grad_frame, axis=-1)
    return np.concatenate((grad_frame, grad_frame, grad_frame), axis=2)   

# Ajuste de Brilho
def ajustarBrilho(frame, beta, code):
    if code == ord('p'):
        beta = (-1) * beta
    
    return cv2.addWeighted(frame, 1, np.zeros(frame.shape, frame.dtype), 0, beta)

# Ajuste de Contraste
def ajustarContraste(frame, alpha, code):
    if code == ord('e'):
        alpha = (-1) * alpha
    
    correction_factor = (259 * (255 + alpha))/(255 * (259 - alpha))
    return cv2.addWeighted(frame, correction_factor, np.zeros(frame.shape, frame.dtype), 0, 0)

# Conversão para Negativo
def converterParaNegativo(frame):
    return 255 - frame

# Conversão para Tons de Cinza
def converterParaCinza(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.expand_dims(frame, axis=-1)
    return np.concatenate((frame, frame, frame), axis=2)

# Redimensionamento
def redimensionarParaMetade(frame):
    scale_percent = 50

    width = int((frame.shape[1] * scale_percent)/100)
    height = int((frame.shape[0] * scale_percent)/100)

    dsize = (width, height)
    return cv2.resize(frame, dsize=dsize)

# Rotação 90 Graus
def rotacionar90Graus(frame):
    return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

# Espelhamento Horizontal
def espelharHorizontalmente(frame):
    return cv2.flip(frame, 1)

# Espelhamento Vertical
def espelharVerticalmente(frame):
    return cv2.flip(frame, 0)

# Captura video e mostra os frames
def cap_and_show():
    camera = 0
    cap = cv2.VideoCapture(camera)

    if(cap.open(camera)):
        while(cap.grab()):
            __, frame = cap.retrieve()

            cv2.imshow("This is you! Smile :)", frame)
            cv2.imshow("Edited image", frame)

            if(cv2.waitKey(1) == 27):
                break

        cap.release()

if __name__ == '__main__':

    main_window = "This is you! Smile :)"
    snd_window = "Edited image"
    ctrl_window = "Control"

    cv2.namedWindow(main_window)
    cv2.namedWindow(snd_window)
    cv2.namedWindow(ctrl_window)

    cv2.createTrackbar('Brightness',
					ctrl_window, 0, 255,
					ajustarBrilho)

    cv2.createTrackbar('Contrast',
					ctrl_window, 0, 127,
					ajustarContraste)
    
    cv2.createTrackbar('Kernel Gaussiano',
					ctrl_window, 3, 20,
					aplicarBorramentoGaussiano)

    camera = 0
    cap = cv2.VideoCapture(camera)
    code_setted = -1

    if(cap.open(camera)):
        recording = False

        while(cap.grab()):
            __, frame1 = cap.retrieve()
            frame2 = frame1.copy()

            code = cv2.waitKey(1)

            if code != -1:
                if code == ord('m'):
                    if not recording:
                        # começa a gravar
                        recording = True
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        video_writer = cv2.VideoWriter("output.avi", fourcc, 20, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                elif code == ord('o'):
                    if recording:
                        video_writer.release()
                        recording = False
                else:
                    code_setted = code            

            if code_setted == ord('g'):
                kernel_size = cv2.getTrackbarPos('Kernel Gaussiano', 'Control')
                frame2 = aplicarBorramentoGaussiano(frame2, kernel_size)

            elif code_setted == ord('c'):
                frame2 = detectarArestas(frame2)

            elif code_setted == ord('s'):
                frame2 = obterEstimativaGradiente(frame2)
            
            elif code_setted == ord('b') or code_setted == ord('p'):
                beta = cv2.getTrackbarPos('Brightness', 'Control')
                frame2 = ajustarBrilho(frame2, beta, code_setted)
            
            elif code_setted == ord('a') or code_setted == ord('e'):
                alpha = cv2.getTrackbarPos('Contrast', 'Control')
                frame2 = ajustarContraste(frame2, alpha, code_setted)
            
            elif code_setted == ord('n'):
                frame2 = converterParaNegativo(frame2)
            
            elif code_setted == ord('l'):
                frame2 = converterParaCinza(frame2)

            elif code_setted == ord('z'):
                frame2 = redimensionarParaMetade(frame2)
            
            elif code_setted == ord('r'):
                frame2 = rotacionar90Graus(frame2)

            elif code_setted == ord('h'):
                frame2 = espelharHorizontalmente(frame2)
            
            elif code_setted == ord('v'):
                frame2 = espelharVerticalmente(frame2)

            elif code_setted == 27:
                break
            
            if recording and code_setted not in (ord('z'), ord('r')):
                video_writer.write(frame2)
            
            cv2.imshow("This is you! Smile :)", frame1)
            cv2.imshow("Edited image", frame2)

        if recording:
            video_writer.release()

        cap.release()