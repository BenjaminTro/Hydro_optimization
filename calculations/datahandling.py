# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 13:47:40 2023

@author: oscar
"""
import pandas as pd
import numpy as np

def read_csv_data(file_path):
    csv_data = pd.read_csv(file_path, parse_dates=[0], dtype=float)
    csv_data.set_index(csv_data.columns[0], inplace=True)
    return csv_data

def read_irr_data(file_path):  
    irr_data = pd.read_csv(file_path, parse_dates=['time'], index_col='time', dtype=float)  
    irr_data.index = irr_data.index.str[:-5]
    irr_data.index = pd.to_datetime(irr_data.index, format='%Y%m%d')
    irr_data['hours'] = irr_data.groupby(irr_data.index.date).cumcount()
    irr_data.index = irr_data.index + pd.to_timedelta(irr_data['hours'], unit='h')
    irr_data.drop(columns='hours', inplace=True)
    return irr_data

def read_excel_data(filename):
    df=pd.read_excel(filename, header=[2])
    return df



