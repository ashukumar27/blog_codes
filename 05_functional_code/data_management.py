#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 12:59:47 2020

@author: ashutosh.k

Data Management: Read Data and Save Data
"""
import config
import pandas as pd

def load_dataset(file_name):
    _data = pd.read_csv(config.DATAPATH + file_name)
    return _data

