##################################################
## This script wil record an screwdriver audio and predict the class label for this on a Raspberry Pi
## Every audio is saved in a different path with an specify metadata for audio control
## Diferent leds turn on depending on the stage you are in
 
## Audio: WAV
## Features generated: MFCC, Chromagram and Mel Spectrogram
## Data: Pickle File
 
## !!!!!!!!!!!!!!!!!!!!!!!!!! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
## !!!!!!!!!!!!!!!!!!!!!!!!!! VERIFY and UNDERSTAND config.yaml BEFORE RUNNING THIS FILE  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
## !!!!!!!!!!!!!!!!!!!!!!!!!! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

##################################################
## Copyright (c) 2023 Robert Bosch TEF4 and its subsidiaries.
## All rights reserved, also regarding any disposal, exploitation, reproduction, editing, distribution,
## as well as in the event of applications for industrial property rights.
##################################################
## Author: Carlos Andres Batres Hermosillo Based 20% on Code from ROG1SLP (SlpP/TEF4)
## Modified by: Carlos Andres Batres Hermosillo (HCM/SlpP TEF4)
## Email: external.Carlos.BatresHermosillo@mx.bosch.com or genaro.rodriguez@mx.bosch.com
## Status: For development reasons only
##################################################

#Libraries to import
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1' #Necesary at the KeyboardInterrupt so you dont get any error
import yaml
from datetime import datetime
from references import ClasesFunciones
from threading import Thread, Lock
from ctypes import *
from contextlib import contextmanager
import pyaudio
import RPi.GPIO as GPIO
import sys

ruta=(sys.path[0])
ruta=str(ruta)[:-4]
ruta2=os.path.join('config','config.yaml')
fileyaml=os.path.join(ruta,ruta2)

#Yaml variables
with open(fileyaml, "r") as f:
    yaml_content = yaml.full_load(f)

NOMBREGRABACION=yaml_content["NomGrabacion"]
LINEA=yaml_content["Linea"]
ESTACION=yaml_content["Estacion"]
AUDIOS=yaml_content["Audios"]
GRABAR=yaml_content["Grabar"]
LISTO=yaml_content["Listo"]
ESPERA=yaml_content["Espera"]
PIEZABUENA=yaml_content["PiezaBuena"]
PIEZAMALA=yaml_content["PiezaMala"]
FRAME_SIZE = yaml_content["Frame_size"]
HOP_LENGTH = yaml_content["Hop_lenght"]
STATION_INDEX=yaml_content["Station_Index"]
FUNCTION_UNIT=yaml_content["Function_Unit"]
WORK_POSITION=yaml_content["Work_Position"]
TOOL_POSITION=yaml_content["Tool_Position"]
TYPE_NUMBER=yaml_content["Type_Number"]
IDENTIFIER=yaml_content["Identifier"]

#Start of main program
if __name__ == '__main__':
    
    path_actual=os.getcwd() #We get the path where the program starts
    proccess = []
    n = 0
    lock=Lock()
    ClasesFunciones.CargaeImagenAudio.setup() #Set up GPIO module on raspberry pi
    GPIO.output(PIEZAMALA,0)
    GPIO.output(PIEZABUENA,0) 

    #Code needed to enable recording with raspberry and avoid warnings and errors
    ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)

    def py_error_handler(filename, line, function, err, fmt):
        pass

    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

    @contextmanager
    def noalsaerr():
        asound = cdll.LoadLibrary('libasound.so')
        asound.snd_lib_error_set_handler(c_error_handler)
        yield
        asound.snd_lib_error_set_handler(None)
    
    #Main loop
    while True: 
        FechaHoraAUDIO=datetime.now()
        year = str(FechaHoraAUDIO.year)
        month = str(FechaHoraAUDIO.month)
        day = str(FechaHoraAUDIO.day)
        if FechaHoraAUDIO.month <= 9:
            month = "0{}".format(FechaHoraAUDIO.month)
        FechaHoraAUDIO=FechaHoraAUDIO.replace(microsecond=0)
        FechaHoraAUDIOFormat=FechaHoraAUDIO.strftime("%Y%m%d_%H%M%S") #Getting the time and date for the metadata
        #Name to save the file
        archivo=LINEA+ESTACION+STATION_INDEX+FUNCTION_UNIT+WORK_POSITION+TOOL_POSITION+"_"+FechaHoraAUDIOFormat+"_"+TYPE_NUMBER+"_"+IDENTIFIER+"_"+NOMBREGRABACION+".wav"
        with noalsaerr():
                audio=pyaudio.PyAudio() #We start pyaudio

        try:
            GPIO.output(LISTO,1)           
            path_clean, base, filee2, path_day=ClasesFunciones.grabar(audio,archivo,year,month,day,LINEA,ESTACION,AUDIOS,path_actual)
            filee3=filee2
            proccess = (ClasesFunciones.hilo("num_{}".format(n),filee3,path_clean,base,path_actual,path_day,lock))
            proccess.start()
            n += 1
            print('Se acabo un ciclo')
            GPIO.output(ESPERA,0)
                 
        except KeyboardInterrupt: #ctrl c to finish the program
            print("Interrupted")
            print('Programa terminado')
            GPIO.output(ESPERA,0)
            GPIO.output(LISTO,0) 
            GPIO.output(GRABAR,0)
            break
