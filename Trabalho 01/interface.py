import io
import os
import laboratory01
import PySimpleGUI as sg
from PIL import Image

class Interface:
    def __init__(self):

        # tema
        sg.theme('Dark Amber')

        # layouts
        layout_images = [
            
            [sg.Image(key="image")],
            [sg.Image(key="imageCopy")]
        ]

        layout_menu = [
            [
                sg.Text("Imagem:"), sg.Input(size=(25,1), key="arquivo"),
                sg.FileBrowse(button_text = "Browse", file_types = (("JPEG files", "*.jpg"), ("PNG files", "*.png"),), key="browsefile"),
                sg.Button("Carregar Imagem")
            ],

            [sg.Text('_'*30)],

            [sg.Text("Escolha qual imagem copiar:")],
            [sg.Radio("De cima", "copy", key="copyLeft", default=True),  sg.Radio("Debaixo", "copy", key="copyRight"), sg.Button("Copiar Imagem")],

            [sg.Text('_'*30)],

            [sg.Text("Escolha qual imagem editar:")],
            [sg.Radio("De cima", "edit", key="editLeft"), sg.Radio("Debaixo", "edit", key="editRight", default=True)],
            [sg.Button("Espelhamento Horizontal")],
            [sg.Button("Espelhamento Vertical")],
            [sg.Button("Luminância")],
            [sg.Button("Quantização"), sg.Slider(range=(0,255), default_value=255, size=(25, 15), orientation='horizontal', font=('Arial', 12), key="colors")],

            [sg.Text('_'*30)],

            [sg.Text("Escolha qual imagem deseja salvar:")],
            [sg.Radio("De cima", "save", key="saveLeft", default=True), sg.Radio("Debaixo", "save", key="saveRight")],
            [sg.Text("Salvar em:"), sg.Input(size=(25,1), key="pastaNome"), sg.FolderBrowse(button_text = "Browse Pasta", key="pasta")],
            [sg.Text("Salvar como:"), sg.Input(size=(25,1), key="nome")],
            [sg.Button("Salvar Imagem JPEG")],
            [sg.Output(size=(30, 1))]
        ]
    

        layout = [[sg.Column(layout_images), sg.VerticalSeparator(), sg.Column(layout_menu)]]
        
        # janela
        self.janela = sg.Window("Photochop").layout(layout)
        

    # extrair dados da tela
    def Iniciar(self):
        image = image_copy = Image.new(mode="RGB", size=(1, 1))

        while True:

            event, values = self.janela.read()
            if event == "Exit" or event == sg.WIN_CLOSED:
                break

            if event == "Carregar Imagem":
                filename = values["arquivo"]
                if os.path.exists(filename):
                    image = laboratory01.carregarImagem(filename)
                    bio = io.BytesIO()
                    image.save(bio, format="PNG")
                    self.janela["image"].update(data=bio.getvalue())

            if event == "Copiar Imagem":
                copyRight = values["copyRight"]
                copyLeft = values["copyLeft"]
                bio = io.BytesIO()

                # como evitar copiar da direita pra esquerda na primeira vez?
                if copyLeft:
                    image_copy = image.copy()
                    image.save(bio, format="PNG")
                elif copyRight:
                    image = image_copy.copy()
                    image_copy.save(bio, format="PNG")
                
                self.janela["image"].update(data=bio.getvalue())
                self.janela["imageCopy"].update(data=bio.getvalue())

            if event == "Espelhamento Horizontal":
                bio = io.BytesIO()

                editRight = values["editRight"]
                editLeft = values["editLeft"]

                if editLeft:
                    image = laboratory01.espelharImagemHorizontal(image)
                    image.save(bio, format="PNG")
                    self.janela["image"].update(data=bio.getvalue())
                    
                elif editRight:
                    image_copy = laboratory01.espelharImagemHorizontal(image_copy)
                    image_copy.save(bio, format="PNG")
                    self.janela["imageCopy"].update(data=bio.getvalue())

            if event == "Espelhamento Vertical":
                bio = io.BytesIO()

                editRight = values["editRight"]
                editLeft = values["editLeft"]

                if editLeft:
                    image = laboratory01.espelharImagemVertical(image)
                    image.save(bio, format="PNG")
                    self.janela["image"].update(data=bio.getvalue())
                    
                elif editRight:
                    image_copy = laboratory01.espelharImagemVertical(image_copy)
                    image_copy.save(bio, format="PNG")
                    self.janela["imageCopy"].update(data=bio.getvalue())

            if event == "Luminância":
                bio = io.BytesIO()

                editRight = values["editRight"]
                editLeft = values["editLeft"]

                if editLeft:
                    image = laboratory01.converterImagemParaCinza(image)
                    image.save(bio, format="PNG")
                    self.janela["image"].update(data=bio.getvalue())
                    
                elif editRight:
                    image_copy = laboratory01.converterImagemParaCinza(image_copy)
                    image_copy.save(bio, format="PNG")
                    self.janela["imageCopy"].update(data=bio.getvalue())
            
            if event == "Quantização":
                bio = io.BytesIO()

                editRight = values["editRight"]
                editLeft = values["editLeft"]
                colors_num = int(values["colors"])

                if editLeft:
                    image = laboratory01.quantizarImagemCinza(image, colors_num)
                    image.save(bio, format="PNG")
                    self.janela["image"].update(data=bio.getvalue())
                    
                elif editRight:
                    image_copy = laboratory01.quantizarImagemCinza(image_copy, colors_num)
                    image_copy.save(bio, format="PNG")
                    self.janela["imageCopy"].update(data=bio.getvalue())
            
            if event == "Salvar Imagem JPEG":
                saveLeft = values["saveLeft"]
                saveRight = values["saveRight"]
                name = values["nome"]
                folderName = values["pastaNome"]

                if saveLeft:
                    laboratory01.salvarImagemJPEG(image, folderName + "/" + name)
                    print("Saved")
                elif saveRight:
                    laboratory01.salvarImagemJPEG(image_copy, folderName + "/" + name)
                    print("Saved")

if __name__ == "__main__":
    interface = Interface()
    interface.Iniciar()