#import matplotlib.pyplot as plt
import plotly.express as px

def plot_irr(data, time_resolution, line_color, start=None, end=None):
    if start is not None and end is not None:
        data = data[start:end]

    data = data.resample(time_resolution).sum()

    fig = px.line(data, x=data.index, y='G(i)', title='Irradiance Data')
    fig.update_xaxes(title='Time')
    fig.update_yaxes(title='Irradiance (W/m^2)')
 
    if line_color:
        fig.update_traces(line=dict(color=line_color))

    fig.show()


def plot_PVcalc(data, time_resolution, line_color, start=None, end=None):
    if start is not None and end is not None:
        data = data[start:end]
    
    data = data.resample(time_resolution).sum()

    fig = px.line(data, x=data.index, y='PV_power', title='PV Power Production')
    fig.update_xaxes(title='Time')
    fig.update_yaxes(title='Power production (MW)')

    if line_color:
        fig.update_traces(line=dict(color=line_color))

    fig.show()


def plot_market(data, time_resolution, line_color, start=None, end=None):
    if start is not None and end is not None:
        data = data[start:end]
    
    data = data.resample(time_resolution).sum()

    fig = px.line(data, x=data.index, y='NO3', title='Market prices (NO3)')
    fig.update_xaxes(title='Time')
    fig.update_yaxes(title='Market price (EUR/MWh)')

    if line_color:
        fig.update_traces(line=dict(color=line_color))

    fig.show()

def plot_load(data, time_resolution, line_color, start=None, end=None):
    if start is not None and end is not None:
        data = data[start:end]
    
    data = data.resample('1T').ffill() #Change 1T to "time_resolution" if we find another load series

    fig = px.line(data, x=data.index, y='Load', title='Load demand')
    fig.update_xaxes(title='Time')
    fig.update_yaxes(title='Load demand (MW)')

    if line_color:
        fig.update_traces(line=dict(color=line_color))

    fig.show()

