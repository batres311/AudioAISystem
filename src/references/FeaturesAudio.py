##################################################
## This script wil contain the fuctions to extract the audio features data
## Here we can extract MFCCs, Mel Spectrogram and Chromagram
 
## Audio: WAV
## Features generated: MFCC, Chromagram and Mel Spectrogram
## Data: Pickle File
 
## The idea behind extracting different features is to test all of them over the same
## machine learning model, and seek which one performs better for the particular use case.
## It could be that Chromagram suits better for an environmental application, but MFCC performs better for industrial scenarios.
 
## !!!!!!!!!!!!!!!!!!!!!!!!!! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
## !!!!!!!!!!!!!!!!!!!!!!!!!! VERIFY and UNDERSTAND config.yaml BEFORE RUNNING THIS FILE  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
import librosa
import pandas as pd

class ARRAYS:
    def __init__(self) -> None:
        pass
    
    def array_MFCC(y,sr,n_mfcc):
        X_sample=librosa.feature.mfcc(y=y, sr=sr,
                            n_mfcc=n_mfcc)
        X_Sample_df= pd.DataFrame(X_sample)
        X_Sample_dfmfcc=X_Sample_df.T

        return X_Sample_dfmfcc

    def arrayMELSPEC(y,sr):
        X_sample=librosa.feature.melspectrogram(y=y, sr=sr)#, nfft=2048
        X_Sample_df = pd.DataFrame(X_sample)
        X_Sample_dfmelspec=X_Sample_df.T
        pd.set_option('display.max_rows', None)

        return X_Sample_dfmelspec

    def arrayCHROMAGRAM(y,sr):
        X_sample=librosa.feature.chroma_cqt(y=y, sr=sr)#, nfft=2048
        X_Sample_df = pd.DataFrame(X_sample)
        X_Sample_df=X_Sample_df.T
        pd.set_option('display.max_rows', None)
        
        return X_Sample_df
