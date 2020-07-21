# -*- coding: utf-8 -*-
"""
XRayDetector
Detector de defectos en imágenes de rayos-x

@author: Noelia Ubierna Fernández
"""

import tkinter as tk
import cv2
from tkinter import filedialog
from os import remove

import os
import numpy as np

import utils
import visualize
from visualize import display_images
import model as modellib
import gdxray

config = gdxray.TrainConfig()
DATA_DIR = os.path.expanduser("~/data/gdxray")


############################################################
#  Dataset
############################################################

class XrayDataset(utils.Dataset):
    """
    Conjunto de datos de las imágenes de rayos-x
    """
    
    def load_gdxray(self, image, width, height): #, annotations, path_mask):
        """ Carga un subconjunto del conjunto de datos.
        image: directorio de la imagen a cargar.
        width: anchura de la imagen.
        height: altura de la imagen.
        """
        self.add_class(source="gdxray", class_id=1, class_name="Casting Defect")

        self.add_image(
            "gdxray",
            image_id=1,
            path=image,
            width=width,
            height=height
        )        
        
        
############################################################
#  Detector
############################################################

class Detector(object):
    """
    Crea la aplicación con sus funciones.
    """
    
    def __init__(self, wind):
        self.image = ""
        self.m = True
        
        self.window = wind
        self.window.config(width=816, height=625)
        self.window.title("XRayDetector")
        
        #Crear un Frame Container
        self.frame = tk.Frame(self.window)
        self.frame.place(x=0, y=0, width=816, height=816)
        
        btnCarga = tk.Button(self.frame, text="Cargar imagen", command = self.get_image)
        btnCarga.place(x=29, y=37, width=365, height=25)
        
        self.btnDetector = tk.Button(self.frame, text="Detectar defectos", command = self.detect_defect)
        self.btnDetector.place(x=402, y=37, width=365, height=25)
        
        self.btnInfo = tk.Button(self.frame, text="Información", command = self.info)
                
        self.btnMask = tk.Button(self.frame, text="Máscaras", command = self.mask)
        
            
    def evaluate_gdxray(self, model, dataset):
        """Evalua la imagen.
        model: modelo con el que se va a evaluar la imagen.
        dataset: imagen a evaluar.
        """
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, 0, use_mini_mask=False)

        # Run object detection
        results = model.detect([image], verbose=0)
        
        # Compute AP
        self.r = results[0]
        visualize.display_instances(image, self.r['rois'], self.r['masks'], self.r['class_ids'], dataset.class_names, self.r['scores'], visualize = False)
        self.show_image()
        
       
    def show_image(self):
        """
        Visualiza la imagen con los defectos.
        """
        error = ""
        
        img = cv2.imread("ImgDetect.png")
        self.width = len(img[0])
        self.height = len(img)
        
        #Redimensionar la imagen
        if self.width > 544:
            resAn = self.width - 544
            resAl = (self.height * resAn) / self.width
            ancho = self.width - resAn
            alto = int(self.height - resAl)
            img = cv2.resize(img, dsize=(ancho, alto))
        
        #Visualizar la imagen
        if cv2.imwrite("ImgCargaDetect.png", img):
            ImgDetect = tk.PhotoImage(file = "ImgCargaDetect.png")
            self.lblImgDetect = tk.Label(self.frame, image = ImgDetect)
            self.lblImgDetect.place(x=136, y=75)
            remove("ImgCargaDetect.png")
        else:
            error = "Error al cargar la imagen."
            
        if error != "":
            self.visualizar_error(error)
        else:
            if self.lblImgCarga.place_info() != {}:
                self.lblImgCarga.place_forget()
        
        self.frame.mainloop()
                
        
    def evaluate(self):
        """Crear el dataset que se va a utilizar, carga la imagen y la evalua.
        """
        dataset_val = XrayDataset()
        dataset_val.load_gdxray(self.image, self.width, self.height)
        dataset_val.prepare()
        
        self.evaluate_gdxray(self.model, dataset_val)
        
        
    def info(self):
        """
        Visualiza la información de los defectos de la imagen evaluada.
        """
        if len(self.r['rois']) != 0:
            info = ""
            for i, f in enumerate(self.r['rois']):
                info = info + "Defecto número " + str(i+1) + ":\n\tBounding boxes: " + str(self.r['rois'][i])
                if self.r['class_ids'][i] == 1:
                    info = info + "\n\tClase: Casting"
                elif self.r['class_ids'][i] == 2:
                    info = info + "\n\tClase: Welding"
                info = info + "\n\tMarca: " + str(self.r['scores'][i]) + "\n"
            
            windInfo = tk.Tk()
            windInfo.title("Información de los defectos")
                
            lblInfo = tk.Label(windInfo, justify = tk.LEFT, text = info)
            lblInfo.place(x=0, y=0)
            lblInfo.pack()
            windInfo.mainloop()
            
        else:
            self.visualizar_error("No hay defectos en la imagen.")
        
        
    def mask(self):
        """
        Visualiza las máscaras de la imagen evaluada
        """
        if len(self.r['masks']) != 0:
            display_images(np.transpose(self.r['masks'], [2, 0, 1]), cmap="Blues_r")
        
        else:
            self.visualizar_error("No hay defectos en la imagen.")
    
    
    def visualizar_error(self, error):
        """
        Visualiza un error en una nueva ventanta.
        error: mensaje a visualizar en la ventana.
        """
        windError = tk.Tk()
        windError.title("Error")
        
        lblErrorCarga = tk.Label(windError, justify = tk.LEFT, text = error, fg = "red")
        lblErrorCarga.place(x=0, y=0)
        lblErrorCarga.pack()
        self.image = ""
        windError.mainloop()
            
            
    def detect_defect(self):
        """
        Carga el modelo que se utlizará para la evaluación y llama a la función evaluate() para evaluar al imagen.
        """
        error = ""
        if self.image != "":
            if self.btnDetector.place_info() != {}:
                self.btnDetector.place_forget()
                
                self.btnInfo.place(x=402, y=37, width=178.5, height=25)
                
                self.btnMask.place(x=588.5, y=37, width=178.5, height=25)
            
            #Comprobar si el modelo ya esta cargado
            if self.m:
                self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
                weights_path = GDXRAY_MODEL_PATH
                
                # Load weights
                self.model.load_weights(weights_path, by_name=True)
                self.m = False
            self.evaluate()
                
        else:
            error = "No hay una imagen cargada."
            
        if error != "":
            self.visualizar_error(error)
            
        self.frame.mainloop()
    
    
    def get_image(self):
        """
        Pide que se seleccione una imagen y la visualiza.
        """
        error = ""
        imagen = filedialog.askopenfilename()
        
        if os.path.exists(imagen):
            if imagen.endswith(".png"):                
                try:
                    img = cv2.imread(imagen, 0)
                    self.width = len(img[0])
                    self.height = len(img)
                    
                    #Redimensionar la imagen
                    if self.width > 544:
                        resAn = self.width - 544
                        resAl = (self.height * resAn) / self.width
                        ancho = self.width - resAn
                        alto = int(self.height - resAl)
                        img = cv2.resize(img, dsize=(ancho, alto))
                    
                    #Visualizar la imagen
                    if cv2.imwrite("ImgCarga.png", img):
                        imgCarga = tk.PhotoImage(file = "ImgCarga.png")
                        self.lblImgCarga = tk.Label(self.frame, image = imgCarga)
                        self.lblImgCarga.place(x=136, y=75)
                        remove("ImgCarga.png")
                        self.image = imagen
                    else:
                        error = "Error al cargar la imagen."
                except:
                    error = "Error al cargar la imagen."
            else:
                error = "El archivo no es una imagen."
        else:
            error = "El directorio no existe."
        
        if error != "":
            self.visualizar_error(error)
        else:
            if self.lblImgDetect.place_info() != {}:
                self.lblImgDetect.place_forget()
                
            if self.btnInfo.place_info() != {}:
                self.btnInfo.place_forget()
                
            if self.btnMask.place_info() != {}:
                self.btnMask.place_forget()
                
            self.btnDetector.place(x=402, y=37, width=365, height=25)

        self.frame.mainloop()
        
        
class InferenceConfig(config.__class__):
    # Ejecute la detección en una imagen a la vez
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    
if __name__ == '__main__':
    ROOT_DIR = os.getcwd()
    MODEL_DIR = os.path.join(ROOT_DIR, "~/logs/gdxray")
    GDXRAY_MODEL_PATH = "C:/Users/Noelia/mask_rcnn_gdxray_0160.h5"

    config = InferenceConfig()
    #config.display()
    
    DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
    TEST_MODE = "inference"
    
    root = tk.Tk()
    app = Detector(root)
    root.mainloop()