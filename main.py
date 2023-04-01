import pandas as pd
import numpy as np
import os
import plotly.express as px

global DIR
DIR = os.getcwd()

def get_fs(df, unit = 'milliseconds'):
    
    # Converting to UNIX time for better calculations
    time = (pd.to_datetime(df.TIME) - pd.Timestamp("1970-01-01")) // pd.Timedelta(1, unit)
    if unit == 'milliseconds':
        k = 10**3
    else:
        raise ValueError(f"Unit: {unit},  must be miliseconds for now")
    
    return 1/np.mean(np.diff(time)/k)

def single_file():
    file = input("Entre o nome do arquivo\n")
    if file in os.listdir(DIR): 
        
        if file.endswith(".csv"):
            
            df = pd.read_csv(os.path.join(DIR,file))
            # Add new column with only time
            df['time'] = pd.to_datetime(df.TIME).dt.strftime('%H:%M:%S')
            print(df['time'])
            
            elementos = list(df.columns)
            grafico = input(f"\n\n{elementos}\n\n\tEscolha dentre as possibilidades acima\n")
            if grafico not in elementos:
                raise ValueError("Valor escolhido nao existente!")
        
            fig = px.line(df, x='time', y=grafico,
                          title = f"<b>{grafico}</b><br><sup>Freq. Amostragem = {get_fs(df):.3f}</sup>")
            fig.show()
            
def main():
    single_file()
            
if __name__ == "__main__":
    main()