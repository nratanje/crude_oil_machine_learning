import numpy as np
import quandl
import pandas as pd
import datetime as dt
from datetime import timedelta
from datetime import datetime
import os

import matplotlib.pyplot as plt
from utility_functions import *

heatmap = False
gtrend_data = True

#================================================================
if heatmap == True:
	#Correlation Heatmap

	train_correlations_df = pd.read_csv("train_correlations.csv", sep=',')
	train_correlations_df.set_index(list(train_correlations_df)[0], inplace=True)

	plt.matshow(train_correlations_df.as_matrix())
	keyword_list = list(train_correlations_df)

	x_pos = np.arange(len(keyword_list))
	plt.xticks(x_pos, keyword_list, rotation='vertical')

	y_pos = np.arange(len(keyword_list))
	plt.yticks(y_pos, keyword_list)

	plt.show()

#================================================================
if gtrend_data == True:
	gtrend_df = pd.read_csv("gtrend_data.csv", sep=',')
	gtrend_df = parse_csv_dates(gtrend_df, '%Y-%m-%d')

	gtrend_df.fillna(method='ffill', inplace=True)
	gtrend_df.fillna(method='bfill', inplace=True)

	#gtrend_df[["strategic petroleum","inflation","oil sands","the oil supply","CAD/JPY","us oil","crude price","crack spread","price of crude","opec","brent","Middle East","Iraq","Gulf War","GDP","Australia","bureau of economic","terrorism","exchange rates","cost per gallon"]].plot(); plt.show()
	
	gtrend_df.plot(); plt.show()