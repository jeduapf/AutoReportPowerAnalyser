import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

global DIR
DIR = os.getcwd()

def add_peaks(fig, time, data, r, c, tipo = 'I'):
    # Get peaks
    if tipo == 'V':
        indices = find_peaks(data, threshold=0.05*np.mean(data))[0]
    else:
        indices = find_peaks(data, threshold=1.05*np.mean(data))[0]
    fig.add_trace(go.Scatter(
    x=[time[j] for j in indices] ,
    y=[data[j] for j in indices],
    mode='markers',
    marker=dict(
        size=9,
        color='black',
        symbol='cross'
    ), showlegend=False,
    name='Detected Peaks'
    ),row=r, col=c)

def check_graph(grafico, elementos):
    if grafico in elementos:
        return True
    else:
        raise ValueError("Valor escolhido nao existente!")
        return False

def input_plot(df, v_nom = 220):
    elementos = list(df.columns)
    df = df.sort_values(by="TIME")
    
    for ele in ['L1','L2','L3']:
        for analysis in ['MAX']:
            
            graphs = [f'VRMS(V) {ele} {analysis}', f'IRMS(A) {ele} {analysis}', f'S(kVA) {ele} {analysis}', f'P(kW) {ele} {analysis}']
            if analysis == 'MAX':
                # Define figure
                fig = make_subplots(rows=4, cols=1, 
                                    vertical_spacing = 0.05, 
                                    shared_xaxes=True, 
                                    subplot_titles= graphs)

                check_graph(graphs[0], elementos)
                fig.add_trace(
                go.Scatter(x = df['TIME'], y = df[graphs[0]], showlegend=False, marker_color='rgba(46,86,241,1)'),
                row=1, col=1
                )
                fig.add_hrect(y0=v_nom*0.95, y1=v_nom*1.05, line_width=0, fillcolor="green", opacity=0.3,row=1, col=1)
                add_peaks(fig, list(df['TIME']), list(df[graphs[0]]),1,1,tipo = 'V')
                
                check_graph(graphs[1], elementos)
                fig.add_trace(
                go.Scatter(x = df['TIME'], y = df[graphs[1]], showlegend=False, marker_color='rgba(46,86,241,1)'),
                row=2, col=1
                )
                add_peaks(fig, list(df['TIME']), list(df[graphs[1]]),2,1)

                check_graph(graphs[2], elementos)
                fig.add_trace(
                go.Scatter(x = df['TIME'], y = df[graphs[2]], showlegend=False,  marker_color='rgba(46,86,241,1)'),
                row=3, col=1
                )
                add_peaks(fig, list(df['TIME']), list(df[graphs[2]]),3,1)

                check_graph(graphs[3], elementos)
                fig.add_trace(
                go.Scatter(x = df['TIME'], y = df[graphs[3]], showlegend=False, marker_color='rgba(46,86,241,1)'),
                row=4, col=1
                )
                add_peaks(fig, list(df['TIME']), list(df[graphs[3]]),4,1)
                
                # edit axis labels
                fig['layout']['yaxis']['title']='Voltagem (V)'
                fig['layout']['yaxis2']['title']='Amperes (A)'
                fig['layout']['yaxis3']['title']='Potencia (kVA)'
                fig['layout']['yaxis4']['title']='Potencia Ativa (kW)'
                
                fig.update_layout(
                autosize=False,
                width=2000,
                height=2500,
                title_text="Análise completa"
                )
                fig.update_yaxes(automargin=True)
                
                # fig.show()
                fig.write_html(f"./{ele}_{analysis}.html")
                fig.write_image(f"./{ele}_{analysis}.pdf", engine="kaleido")
    
def main():
    
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
        try:
            file = next(iteration_files)
        except:
            file = False
            
    input_plot(df)
    
if __name__ == "__main__":
    main()