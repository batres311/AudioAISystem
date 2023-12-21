##################################################
## This script wil contain the classes and fuctions for recording screwdrivers audios and predict labels for these
## Here we can find functions to create threads, recording, generating pickle files or create paths for saving
## Diferent leds turn on depending on the phase you are in
 
## Audio: WAV
## Features generated: MFCC, Chromagram and Mel Spectrogram
## Data: Pickle File
 
## The idea behind generating threads is that we can carry out different processes at the same time
## so the code is not read only sequentially this allows us to make recordings while other processes
## for extracting audio characteristics are carried out...
 
## !!!!!!!!!!!!!!!!!!!!!!!!!! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
## !!!!!!!!!!!!!!!!!!!!!!!!!! VERIFY and UNDERSTAND config.yaml BEFORE RUNNING THIS FILE  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
## !!!!!!!!!!!!!!!!!!!!!!!!!! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 
##################################################
## Copyright (c) 2023 Robert Bosch TEF4 and its subsidiaries.
## All rights reserved, also regarding any disposal, exploitation, reproduction, editing, distribution,
## as well as in the event of applications for industrial property rights.
 
##################################################
## Author: Carlos Andres Batres Hermosillo Based 30% on Code from ROG1SLP (SlpP/TEF4)
## Modified by: Carlos Andres Batres Hermosillo (HCM/SlpP TEF4)
## Email: external.Carlos.BatresHermosillo@mx.bosch.com or genaro.rodriguez@mx.bosch.com
## Status: For development reasons only
##################################################

#Libraries to import
import numpy as np
import librosa
import os
import yaml
import pandas as pd
import pyaudio #Library that helps to obtain the audio and format it
import wave  #Allows reading and writing wav files
import scipy.io.wavfile as waves #Important library for audio data
import shutil #Library to move files to different folders
from pydub import AudioSegment
import time
import threading
import pickle
import concurrent.futures
from references import FeaturesAudio
import RPi.GPIO as GPIO
import tensorflow
import sys

ruta=(sys.path[0])
ruta=str(ruta)[:-4]
ruta2=os.path.join('config','config.yaml')
fileyaml=os.path.join(ruta,ruta2)

#Yaml variables
with open(fileyaml, "r") as f:
    yaml_content = yaml.full_load(f)


patharraysmfcc=yaml_content["PATHSARRAYS"]["PATHMFCC"]
patharrayschro=yaml_content["PATHSARRAYS"]["PATHCHRO"]
patharraysmelspec=yaml_content["PATHSARRAYS"]["PATHMELSPEC"]
FRAME_RATE = yaml_content["Frame_rate"]
CHANNELS = yaml_content["Channels"]
FRAMESPERBUFFER= yaml_content["FramesPerBuffer"]
FRAME_SIZE = yaml_content["Frame_size"]
N_MFCCs= yaml_content["Number_MFCCs"]
RAW= yaml_content["Raw"]
THRESHOLD1= yaml_content["Threshold1"]
THRESHOLD2= yaml_content["Threshold2"]
CLEAN = yaml_content["Clean"]
DATA=yaml_content["Data"]
PRODUCTO=yaml_content["Producto"]
DURACION=yaml_content["duracion"]
STARTSEC= yaml_content["StartSec"]
ENDSEC= yaml_content["EndSec"]
Empezar = yaml_content["BotonGrabar"]
Detener = yaml_content["BotonDetener"]
GRABAR=yaml_content["Grabar"]
LISTO=yaml_content["Listo"]
ESPERA=yaml_content["Espera"]
PICKLEREADY=yaml_content["PickleReady"]
PIEZABUENA=yaml_content["PiezaBuena"]
PIEZAMALA=yaml_content["PiezaMala"]
MODELO=yaml_content["Modelo"]
CARPETAMODELOS=yaml_content["CarpetaModelos"]


class CargaeImagenAudio():
    def __init__(self,audio,archivo,year,month,day,linea,estacion,audios,path_programa,):
       
        self.audio=audio
        self.archivo=archivo
        self.year=year
        self.month=month
        self.day=day
        self.linea=linea
        self.estacion=estacion
        self.audios=audios
        self.path_programa=path_programa

    def GuardaAudio(self):
        if not os.path.exists(DATA):
            os.makedirs(DATA, exist_ok=True)
        path_base=os.path.join(self.path_programa,DATA)
        path_producto=os.path.join(path_base,PRODUCTO)
        path_linea=os.path.join(path_producto,self.linea)
        path_estacion=os.path.join(path_linea,self.estacion)
        path_trabajo=os.path.join(path_estacion,self.audios)
        path_year = os.path.join(path_trabajo, self.year)
        path_month = os.path.join(path_year, self.month)
        path_day = os.path.join(path_month, self.day)
        path_raw=os.path.join(path_day,RAW)
        path_clean=os.path.join(path_day,CLEAN)
       
        if os.path.isdir(path_base) == True: #If the folder does not exist then create the folder
            if os.path.isdir(path_producto) == False:
                os.mkdir(path_producto)
            if os.path.isdir(path_producto) == True:
                if os.path.isdir(path_linea) == False:
                    os.mkdir(path_linea)
                if os.path.isdir(path_linea) == True:
                    if os.path.isdir(path_estacion) == False:
                        os.mkdir(path_estacion)
                    if os.path.isdir(path_estacion) == True:       
                            if os.path.isdir(path_trabajo) == False:
                                os.mkdir(path_trabajo)
                            if os.path.isdir(path_trabajo)==True:
                                if os.path.isdir(path_year)==False:
                                    os.mkdir(path_year)
                                if os.path.isdir(path_year) == True: 
                                    if os.path.isdir(path_month) == False:
                                        os.mkdir(path_month)
                                    if os.path.isdir(path_month)==True:
                                        if os.path.isdir(path_day)==False:
                                            os.mkdir(path_day)
                                        if os.path.isdir(path_day)==True:
                                            if os.path.isdir(path_raw)==False:
                                                os.mkdir(path_raw)
                                        if os.path.isdir(path_day)==True:
                                            if os.path.isdir(path_clean)==False:
                                                os.mkdir(path_clean)
 
                                        return path_raw,path_clean,path_day
    @staticmethod
    def setup():
        GPIO.setwarnings(False) 
        GPIO.setmode(GPIO.BOARD)       # Numbers GPIOs by physical location
        # Set Led Pins mode to output
        # Set button Pin mode to input
        GPIO.setup(Detener, GPIO.IN, pull_up_down=GPIO.PUD_UP)      
        GPIO.setup(Empezar, GPIO.IN, pull_up_down=GPIO.PUD_UP) 
        GPIO.setup(GRABAR, GPIO.OUT)
        GPIO.setup(LISTO, GPIO.OUT)
        GPIO.setup(ESPERA, GPIO.OUT)
        GPIO.setup(PIEZABUENA, GPIO.OUT)
        GPIO.setup(PIEZAMALA, GPIO.OUT)
        GPIO.setup(PICKLEREADY, GPIO.OUT) 

    #Envelope functions to eliminate noise using amplitude                                 
    def envelope(self,y,rate,threshold):
        mask = []
        y = pd.Series(y).apply(np.abs)
        y_mean = y.rolling(window=int(rate/100), min_periods=1, center=True).mean()
        for mean in y_mean:
            if mean > threshold:
                mask.append(True)
            else:
                mask.append(False)
        return np.array(mask)
    
    def envelope2(self,y,rate,threshold):
        mask = []
        y = pd.Series(y).apply(np.abs)
        y_mean = y.rolling(window=int(rate/100), min_periods=1, center=True).mean()
        for mean in y_mean:
            if mean < threshold:
                mask.append(True)
            else:
                mask.append(False)
        return np.array(mask)
    
    #Loop for recording 
    def loop(self):
        while True:
            if GPIO.input(Empezar)==0:  
                #Current or flow opens
                stream=self.audio.open(format=pyaudio.paInt16,channels=CHANNELS,
                                    rate=FRAME_RATE,input=True, #Sample rate 44.1KHz
                                    frames_per_buffer=FRAME_SIZE) 
                GPIO.output(LISTO,0)
                time.sleep(0.2) 
                GPIO.output(GRABAR,1)                
                print("Grabando ...") #Message that recording has started
                frames=[] #Here we save the recording
                for i in range(0,int(FRAME_RATE/FRAMESPERBUFFER*DURACION)):
                    data=stream.read(FRAMESPERBUFFER)
                    frames.append(data)  

                stream.stop_stream()    #Close recording
                stream.close()          #Stop stream
                self.audio.terminate()
                path_raw,path_clean,path_day=self.GuardaAudio()
                waveFile=wave.open(self.archivo,'wb') #Our file is created
                waveFile.setnchannels(CHANNELS) #Design channels
                waveFile.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
                waveFile.setframerate(FRAME_RATE) #Sample rate is passed
                waveFile.writeframes(b''.join(frames))
                waveFile.close() #Close the file
                GPIO.output(GRABAR,0)
                time.sleep(0.2) 
                return path_raw, path_clean, path_day

    
    def AcomodoPathRAW(self,path_espacio):
        shutil.move(self.archivo, path_espacio)
        path_final=os.path.join(path_espacio,self.archivo)
        return path_final
    
    @staticmethod
    def AcomodoPathClean(path_espacio,extract,archivo):
        path_final=os.path.join(path_espacio,archivo)
        extract.export(path_final, format="wav")
        os.chdir(path_espacio)

    @staticmethod
    def LoadAudio_Turn2Decibels(archivo):
        y, sr = librosa.load(archivo, sr=FRAME_RATE) 
        D = librosa.stft(y) 
        # STFT of y 
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max) 
        #, ref=np.max
        return y,S_db,sr
    
    @staticmethod
    def RecorteAudio(file): # Function to trim the audio
        sound = AudioSegment.from_file(file=file,format="wav")
        StartTime=STARTSEC*1000
        EndTime=ENDSEC *1000
        extract=sound[StartTime:EndTime]
        return extract
        
    @staticmethod
    def SavePickle(direccion,array): # Function to save pickles
        with open(direccion, 'wb') as handle:
            pickle.dump(np.c_[array], handle, protocol=2)   # 2 is compatible to all python versions
            print("Pickle updated successfully")
            return True
        
#Class to use and generate threads
class hilo:
    def __init__(self,name,signal3,path_clean,base,path_actual,path_day,lock) -> None:
        self.name = name
        self.signal3 = signal3
        self.path_clean = path_clean
        self.base = base
        self.path_actual=path_actual
        self.path_day=path_day
        self.lock=lock
        self.t1 = threading.Thread(target=sum,args=(self.name,self.signal3,self.path_clean,self.base,self.path_actual,self.path_day,self.lock))#, 
        
    def start(self):
        self.t1.start()

def removeenvfiles(): #Function to delete files created when we extract the envelope
    if os.path.exists("clean"+"Grab"+".wav")==True:
        os.remove("clean"+"Grab"+".wav")  
    if os.path.exists("New"+"clean"+"Grab"+".wav")==True:
        os.remove("New"+"clean"+"Grab"+".wav")

def sum(name,signal3,path_clean,base,path_actual,path_day,lock):#
    try: #Multiplicar frame rate por el endsec
        lock.acquire()
        extract=CargaeImagenAudio.RecorteAudio(signal3)
        CargaeImagenAudio.AcomodoPathClean(path_clean,extract,base)
        y,S_db,sr=CargaeImagenAudio.LoadAudio_Turn2Decibels(base)#"newfile"+str(i)+".wav"
        os.chdir(path_actual)
        removeenvfiles()
        lock.release()

        # Generate threads to obtain every audio feature separately
        with concurrent.futures.ThreadPoolExecutor() as executor:
            t1=executor.submit(FeaturesAudio.ARRAYS.array_MFCC,y,sr,N_MFCCs)
            t2=executor.submit(FeaturesAudio.ARRAYS.arrayMELSPEC,y,sr)
            t3=executor.submit(FeaturesAudio.ARRAYS.arrayCHROMAGRAM,y,sr)
        mfcc=t1.result()
        melspec=t2.result()
        chromagram=t3.result()
        os.chdir(path_day)

        #Creating paths to store diferent audio features
        if not os.path.exists(patharraysmfcc): 
            os.makedirs(patharraysmfcc, exist_ok=True) 
        if not os.path.exists(patharraysmelspec): 
            os.makedirs(patharraysmelspec, exist_ok=True) 
        if not os.path.exists(patharrayschro): 
            os.makedirs(patharrayschro, exist_ok=True) 
        audio_filename=base

        array_filename_to_save = str(audio_filename).replace(".wav", "_", 1)
        path1=os.path.join(path_day,patharraysmfcc)
        path2=os.path.join(path_day,patharraysmelspec)
        path3=os.path.join(path_day,patharrayschro)
        os.chdir(path_actual)
        archivo2=os.path.join(path1,array_filename_to_save+'arraysmfcc.p')
        archivo3=os.path.join(path2,array_filename_to_save+'arraysmelspec.p')
        archivo4=os.path.join(path3,array_filename_to_save+'arrayschro.p')

        #Saving arrays as pickle files
        CargaeImagenAudio.SavePickle(archivo2,mfcc)
        CargaeImagenAudio.SavePickle(archivo3,melspec)
        CargaeImagenAudio.SavePickle(archivo4,chromagram)
        print(f'{name}')
       
        del signal3
        prediccion=[]
        pathtft=os.path.join(ruta,CARPETAMODELOS)
        with open(archivo2, 'rb') as l:
            prediccion= pickle.load(l)
        X_to_predict =np.array(prediccion)
        # Load the TFLite model
        #AI model to record 2 seconds audios
        #model_path = os.path.join(pathtft,'modelocategoricobuenosanitycheck2.tflite')  # Update with the path to your .tflite model
        #AI model to record 1 second audios
        model_path = os.path.join(pathtft,MODELO) 
        interpreter = tensorflow.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        tensor_details = interpreter.get_tensor_details()
        input_tensor_index = input_details[0]['index']
        X_to_predict = np.expand_dims(X_to_predict, axis=0)
        X_to_predict = np.expand_dims(X_to_predict, axis=3)
        interpreter.set_tensor(input_tensor_index, X_to_predict)
        
        #Get prediction
        interpreter.invoke()
        yhat =interpreter.get_tensor(output_details[0]['index'])
        predicted_class_indices=np.argmax(yhat,axis=1)
        prediction_class=predicted_class_indices[0]

        #Print pedictions and turn on the corresponding led
        if prediction_class==0: #Bosch screwdriver class
            GPIO.output(PIEZABUENA,1)
            time.sleep(2) 
            GPIO.output(PIEZABUENA,0)

        if prediction_class==1: #Black & Decker screwdriver class
            GPIO.output(PIEZAMALA,1)
            time.sleep(2) 
            GPIO.output(PIEZAMALA,0)

        if prediction_class==2:  #Ambient noise class
            GPIO.output(PICKLEREADY,1)
            time.sleep(2)
            GPIO.output(PICKLEREADY,0)
        print(f'Predicted class is: ', prediction_class)
        
    except:
        print("Grabe un audio de mas de 1 segundo y vuelva a usar el programa") 
        os.chdir(path_actual)
        removeenvfiles()
        for i in range (5):
            GPIO.output(PICKLEREADY,1)
            time.sleep(0.25)
            GPIO.output(PICKLEREADY,0)
            time.sleep(0.25)
            i+=1 

def grabar(audio,archivo,year,month,day,LINEA,ESTACION,AUDIOS,path_actual):
    print("Listo para grabar presiona g")
    Grab1=CargaeImagenAudio(audio,archivo,year,month,day,LINEA,ESTACION,AUDIOS,path_actual)
    path_raw,path_clean,path_day=Grab1.loop()	
    print("La grabacion ha terminado ") #End of recording message
    GPIO.output(ESPERA,1)

    signal,S_db1,sample_rate=CargaeImagenAudio.LoadAudio_Turn2Decibels(archivo)
    base= archivo  
    path_final=Grab1.AcomodoPathRAW(path_raw)#Audiobueno_path_export,path_actual,

    # Functions for envelope
    # mask =Grab1.envelope(signal,sample_rate,THRESHOLD1)#Bosch=0.004,0.0025 b0.003 0.005 0.006 0.007
    # waves.write(filename="clean"+"Grab"+".wav", rate=sample_rate, data=signal[mask])
    # filee="clean"+"Grab"+".wav"
    # signal1, rate1 = librosa.load("clean"+"Grab"+".wav", sr=FRAME_RATE)
    # mask2 = Grab1.envelope2(signal1,rate1,THRESHOLD2)#Bosch=0.0095,0.0097  b0.016 0.017 0.018 0.019
    waves.write(filename="New"+"clean"+"Grab"+".wav", rate=sample_rate, data=signal)
    filee2="New"+"clean"+"Grab"+".wav"

    return path_clean, base, filee2, path_day
