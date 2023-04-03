import pandas as pd
import numpy as np
import os

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
from graphs import input_plot, power_peaks_plot, phase_balance_plot,current_peaks_plot

global DIR
DIR = os.getcwd()
DIR_SvESI = "C:/Users/jedua/OneDrive/Documents/Codes/Personal/MAR_722/Data/SESI"

def generate_df(DIR = DIR):
    '''
        From all files in the current folder join all them in a single Pandas Dataframe
        
        Inputs:
            DIR: Directory where all files to be concatenated are. Usually the current directory.
        
        Outputs:
            df: Pandas dataframe containing the concatenation of all files in the current filder
        
    '''
    iteration_files = iter(os.listdir(DIR))
    
    # Join files in one plot
    file = next(iteration_files)
    flag = True

    while file:
        # If it is the first file just open it 
        if flag:
            if file.endswith(".csv"):
                flag = False
                df = pd.read_csv(os.path.join(DIR,file))
                
        # Else append new file data to the dataframe
        else:
            if file.endswith(".csv"):
                df_aux = pd.read_csv(os.path.join(DIR,file))
                df = pd.concat([df,df_aux])
                
        # Check if there is another file to append
        try:
            file = next(iteration_files)
        except:
            file = False
            
    return df

def main():
    
    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-IMPORTANT*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    # Change Directory to DATA for all data
    # *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-IMPORTANT*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    df = generate_df(DIR = "C:/Users/jedua/OneDrive/Documents/Codes/Personal/MAR_722/Data/SESI")   
    
    # #
    # # POWER ANALYSIS
    # #
    # print('\n\n\t Iniciando análise de potência...')
    # power = os.path.join(DIR, 'power_analysis')
    # try: 
    #     os.mkdir(power) 
    # except OSError as error: 
    #     print(error)  
    # power_peaks_plot(df, v_nom = 220, dir = power.replace('\\','/') + '/')
    # power_peaks_plot(df, v_nom = 220, dir = DIR)
    
    #
    # CURRENT ANALYSIS
    #
    print('\n\n\t Iniciando análise de corrente...')
    current = os.path.join(DIR, 'current_analysis')
    try: 
        os.mkdir(current) 
    except OSError as error: 
        print(error)  
    current_peaks_plot(df, v_nom = 220, dir = current.replace('\\','/') + '/')   
       
    #
    # BALANCE ANALYSIS
    #
    print('\n\n\t Iniciando balanço de fase...\n')
    phase = os.path.join(DIR, 'phase_balance_n_pf')
    try: 
        os.mkdir(phase) 
    except OSError as error: 
        print(error)  
    phase_balance_plot(df, dir = phase.replace('\\','/') + '/')
    
if __name__ == "__main__":
    main()