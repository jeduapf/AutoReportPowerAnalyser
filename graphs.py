import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

# TODO: Arrumar graficos e criar graficos de recortes de picos de potencia ( janelas para relatorio )
# TODO: Criar arquivos com graficos completos V, I, P, S
# TODO: Criar arquivos com graficos especificos 
# TODO: Criar arquivo com grafios de balanceamento

global peak_sensibility,smooth_per
smooth_per = 0.8
peak_sensibility = 1.05

def add_peaks(fig, time, data, r, c, tipo = 'I', top_peaks = 3):
    '''
        Add visual peaks to the power data graphs
        
        Inputs:
            fig: Plotly figure with the data
            time: List of timestamp values of the collected data
            data: List of values of the current data to be added peaks
            r: Row of the subplot to be added the peaks 
            c: Column of the subplot to be added the peaks
            tipo: If type is V (tension) then it shouldn't surpass 5% of the mean. Else it shouldn't surpass 120% of the mean.
            top_peaks: Int that gets the most top peaks found
            
        Output:
            List of top peaks detected
            "Add trace to existing plotly figure"
    '''

    # Get peaks
    if tipo == 'V':
        indices = find_peaks(data, threshold=0.05*np.mean(data))[0]
    elif tipo =='IRMS':
        indices = find_peaks(data, prominence=np.mean(data))[0]
    elif tipo =='P':
        indices = find_peaks(data, prominence=np.mean(data))[0]
    elif tipo =='Q' or tipo == 'S':
        indices = find_peaks(data, prominence=np.mean(data))[0]
    else:
        indices = find_peaks(data, prominence=10)[0]

        
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

    list_peaks = [(time[j],data[j]) for j in indices]
    list_peaks.sort(key=lambda a: a[1], reverse=True)

    if len(indices) > top_peaks:
        return list_peaks[0:top_peaks],None
    else:
        return list_peaks,len(indices)

def check_graph(grafico, elementos):
    '''
        Verify if the graph to be ploted exist between the data acquired
        
        Input:
            grafico: String containing the name of the data to be ploted
            elementos: List of Strings containing all data acquired from the power meter
        
        Output:
            Bool: True if graph exist in element, false if not
    '''
    if grafico in elementos:
        return True
    else:
        raise ValueError("Valor escolhido nao existente!")
        return False
                
def get_ylabel_pt_br(unit):

    if unit == 'V':
        return 'Voltagem (V)'
    elif unit == 'A':
        return 'Amperes (A)'
    elif unit == 'kVA':
        return 'Potencia (kVA)'
    elif unit == 'kW':
        return 'Potencia Ativa (kW)'
    elif unit == 'kvar':
        return 'Potencia Reativa (kVAr)'
    else:
        raise ValueError("Unit not found")

def add_peak_annotation(fig, k, graphs, list_peaks, top_peaks):
    TEXT =  '''
            <b>Top Picos:</b><br>
            '''
    if list_peaks is not None:
        for count in range(top_peaks):
            TEXT += f'{list_peaks[count][0]} -> {list_peaks[count][1]}<br>'
            
        fig.add_annotation(text=TEXT,
                            xref="paper", yref="paper",
                            x=0.0, y=1-k*(1/len(graphs) + 0.025), showarrow=False)
    
def add_subplots_peaks(df, fig, graphs, elementos, v_nom = 220, top_peaks = 3):
    
    # For each tipe of graph in the list of graphs in the report add subplot
    for k in range(len(graphs)):
        # Get unit of the graph
        unit = graphs[k].split(')')[0].split('(')[-1]

        
        check_graph(graphs[k], elementos)
        fig.add_trace(
        go.Scatter(x = df['TIME'], y = df[graphs[k]], showlegend=False, marker_color='rgba(46,86,241,1)'),
        row=k+1, col=1
        )
        per = int(smooth_per*len(df[graphs[k]])/100)
        
        fig.add_trace(
        go.Scatter(x=df['TIME'], y=df[graphs[k]].rolling(per).mean(), line=dict(color='red', width=3), name=f"Media movel {per} pontos"), # Mean average
        row=k+1, col=1
        )
        
        
        # If it is a Voltage graph add green safety tension region
        if unit == 'V':
            fig.add_hrect(y0=v_nom*0.95, y1=v_nom*1.05, line_width=0, fillcolor="green", opacity=0.3,row=k, col=1)
            list_peaks,size = add_peaks(fig, list(df['TIME']), list(df[graphs[k]]),k+1,1,tipo = 'V', top_peaks = 3)
            if size is None:
                add_peak_annotation(fig, k, graphs, list_peaks, top_peaks)
            else: 
                add_peak_annotation(fig, k, graphs, list_peaks, size)
            
        else:
            if unit != 'kvar':

                list_peaks,size = add_peaks(fig, list(df['TIME']), list(df[graphs[k]]),k+1,1, tipo = graphs[k].split('(')[0], top_peaks = 3)
                if size is None:
                    add_peak_annotation(fig, k, graphs, list_peaks, top_peaks)
                else: 
                    add_peak_annotation(fig, k, graphs, list_peaks, size)
                
        # edit axis labels
        if k == 0:
            fig['layout']['yaxis']['title']=get_ylabel_pt_br(unit)
        else:
            fig['layout'][f'yaxis{k+1}']['title']=get_ylabel_pt_br(unit)
        
    return fig

def power_peaks_plot(df, v_nom = 220, dir = './'):
    
    elementos = list(df.columns)
    df = df.sort_values(by="TIME")
    
    # For each of the phases plot the graphs
    for ele in ['L1','L2','L3','ALL']:
        
        # Which type of analysis (MAX/MIN/AVG)
        for analysis in ['MAX']:
            
            # List of graphs to be ploted in a single HTML
            graphs = [f'P(kW) {ele} {analysis}', f'S(kVA) {ele} {analysis}', f'Q(kvar) {ele} {analysis}']
            
            # TODO: Acho que da na mesma nao tem diferenca MAX/MIN/AVG pro grafico...
            if analysis == 'MAX':
                
                # Define figure
                fig = make_subplots(rows=len(graphs), cols=1, 
                                    vertical_spacing = 0.05, 
                                    shared_xaxes=True, 
                                    subplot_titles= graphs)
                
                fig = add_subplots_peaks(df, fig, graphs, elementos, v_nom = 220)
                
                fig.update_layout(
                autosize=False,
                width=2000,
                height=500*len(graphs)+400,
                title = {
                        'text': f"Análise completa picos de potência ({ele}-{analysis})",
                        'y':0.99, 
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                        }
                )
                fig.update_yaxes(automargin=True)
                
                # fig.show()
                fig.write_html(dir + f"{ele}_{analysis}.html")
                # fig.write_image(f"./{ele}_{analysis}.pdf", engine="kaleido")

def get_unit(ele):
    if ele == 'P':
        return 'kW'
    elif ele == 'Q':
        return 'kvar'
    elif ele == 'S':
        return 'kVA'
                
def phase_balance_plot(df, dir = './'):
    
    elementos = list(df.columns)
    df = df.sort_values(by="TIME")
    
    bar_dict = {}
    # Which type of analysis (MAX/MIN/AVG)
    for analysis in ['MAX']:
        
        # For each of the phases plot the graphs
        for ele in ['P','S']:
            
            # List of graphs to be ploted in a single HTML
            graphs = [f'{ele}({get_unit(ele)}) L1 {analysis}', f'{ele}({get_unit(ele)}) L2 {analysis}', f'{ele}({get_unit(ele)}) L3 {analysis}']
            
            L1 = np.array(df[graphs[0]])
            L2 = np.array(df[graphs[1]])
            L3 = np.array(df[graphs[2]])

            s = (L1+L2+L3)
            s[s == 0] = np.nan
            
            if ele == 'P':
                sum_p = s
            elif ele == 'S':
                sum_s = s
            
            perc_L1 = np.divide(L1,s)
            perc_L2 = np.divide(L2,s)
            perc_L3 = np.divide(L3,s)
            
            perc_L1[perc_L1 == np.nan] = 0
            perc_L2[perc_L2 == np.nan] = 0
            perc_L3[perc_L3 == np.nan] = 0
            
            final_L1 = 100*np.mean(perc_L1)
            final_L2 = 100*np.mean(perc_L2)
            final_L3 = 100*np.mean(perc_L3)
            
            bar_dict[f'{ele}({get_unit(ele)})'] = [final_L1,final_L2,final_L3]
        
        # Power factor figure
        fp = np.abs(sum_p/sum_s)
        fp[fp == np.nan] = 0
        mean_fp = np.mean(fp)
        fig_fp = px.line(x = df['TIME'], y = fp, title="Fator de potência instantâneo")
        fig_fp.update_layout(xaxis_title="Tempo (s)",
                             yaxis_title="Fator de potência (0-1)")
        fig_fp.write_html(dir + f"fp_{analysis}.html")
        
        # Balance figure
        x = ['P(kW)', 'S(kVA)']
        y = []
        for count in range(len(graphs)):
            y.append([bar_dict['P(kW)'][count],bar_dict['S(kVA)'][count]])
        
        fig = go.Figure(go.Bar(x = x, y = y[0], name='L1'))
        fig.add_trace(go.Bar(x = x,y = y[1], name = 'L2'))
        fig.add_trace(go.Bar(x = x,y = y[2], name = 'L3'))
        fig.update_layout(  barmode="relative",
                            title={
                                    'text': "Balanço de fases",
                                    'y':0.95,
                                    'x':0.5,
                                    'xanchor': 'center',
                                    'yanchor': 'top'},
                            xaxis_title="Unidade",
                            yaxis_title="Porcentagem",
                            legend_title="Fase",
                            )
        fig.add_annotation(text=f'fator de potência médio:<br><b>{mean_fp:.03f}</b>',
                        xref="paper", yref="paper",
                        x=0.5, y=0.55, showarrow=False)
        # fig.show()
        fig.write_html(dir + f"balance_{analysis}.html")

def current_peaks_plot(df, v_nom = 220, dir = './'):
    
    elementos = list(df.columns)
    df = df.sort_values(by="TIME")
        
    # Which type of analysis (MAX/MIN/AVG)
    for analysis in ['MAX']:
        
        # List of graphs to be ploted in a single HTML
        graphs = [f'IRMS(A) L1 {analysis}', f'IRMS(A) L2 {analysis}', f'IRMS(A) L3 {analysis}', f'IRMS(A) N {analysis}']
        
        # TODO: Acho que da na mesma nao tem diferenca MAX/MIN/AVG pro grafico...
        if analysis == 'MAX':
            
            # Define figure
            fig = make_subplots(rows=len(graphs), cols=1, 
                                vertical_spacing = 0.05, 
                                shared_xaxes=True, 
                                subplot_titles= graphs)
            
            fig = add_subplots_peaks(df, fig, graphs, elementos, v_nom = 220)
            
            fig.update_layout(
            autosize=False,
            width=2000,
            height=500*len(graphs)+400,
            title = {
                    'text': f"Análise completa picos de corrente ({analysis})",
                    'y':0.99, 
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                    }
            )
            fig.update_yaxes(automargin=True)
            
            # fig.show()
            fig.write_html(dir + f"current_{analysis}.html")
            # fig.write_image(f"./{ele}_{analysis}.pdf", engine="kaleido")

# DEPRECATED
def input_plot(df, v_nom = 220):
    elementos = list(df.columns)
    df = df.sort_values(by="TIME")
    
    # For each of the phases plot the graphs
    for ele in ['L1','L2','L3']:
        
        # Which type of analysis (MAX/MIN/AVG)
        for analysis in ['MAX']:
            
            # List of graphs to be ploted in a single HTML
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
                # fig.write_image(f"./{ele}_{analysis}.pdf", engine="kaleido")
