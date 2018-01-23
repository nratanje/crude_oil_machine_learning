import numpy as np
import quandl
import pandas as pd
import datetime as dt
from datetime import timedelta
from datetime import datetime
import os

import matplotlib.pyplot as plt

#print(plt.style.available)
plt.style.use('seaborn-ticks')

from utility_functions import *
from acquire_data import *

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor #ExtraTreesClassifier

#================================================================

get_quandl_data = False
get_gtrend_data = False

crude_oil_type = "BRENT"#OPEC"#BRENT WTI

#pre_start_time = dt.datetime(2017, 9, 1)
#start_time = dt.datetime(2017, 10, 1)
#end_time = dt.datetime(2018, 1, 2)

pre_start_time = dt.datetime(2003, 12, 1)
start_time = dt.datetime(2004, 1, 1)
end_time = dt.datetime(2018, 1, 2)

test_to_train_ratio = 0.5
use_daily_data = False

use_quandl_returns = False
use_quandl_hist_vol = False

use_gtrend_returns = False
#use_gtrend_hist_vol = False

feature_selection = True
preprocess = True

low_var_mult = 0.000001
high_corr_mult = 0#1#2

estimators = 100
feature_number= 20

lm_ply_deg = 1

#================================================================

if get_quandl_data == True:
	download_quandl_data(pre_start_time, end_time)
	
if get_gtrend_data == True:
	print
	gtrend_searchwords_list = pd.read_csv("gtrend_searchwords.csv", header=None).to_dict(orient='list')[0]
	download_gtrend_data(start_time, end_time, gtrend_searchwords_list)

#================================================================
print "Processing quandl data."

#Read downloaded .csv files:
ps_quandl_df = pd.read_csv("crude_oil.csv", sep=',')
ps_quandl_df.rename(columns={"Date": "date"}, inplace=True)
ps_quandl_df = parse_csv_dates(ps_quandl_df, '%Y-%m-%d')

#Restrict Quandl data to only one crude oil type, replace empty values with last value to calculate returns
ps_quandl_s = ps_quandl_df[crude_oil_type]
ps_quandl_s.fillna(method='ffill', inplace=True)

if use_daily_data == False:
	ps_quandl_s = ps_quandl_s.resample('MS').mean(); #avg_quandl_s.plot(); plt.show()

#Calculate simple returns and a rolling variance for 21 prior entries:
#f = lambda x: 100*x
smp_rtn_ps_quandl_s = ps_quandl_s.pct_change().fillna(0)#.apply(f)

hv_smp_rtn_quandl_s = smp_rtn_ps_quandl_s.rolling(window=21, min_periods=1).var()
hv_smp_rtn_quandl_s = hv_smp_rtn_quandl_s.drop(hv_smp_rtn_quandl_s[:start_time].index)

smp_rtn_quandl_s = smp_rtn_ps_quandl_s.drop(smp_rtn_ps_quandl_s[:start_time].index)
del smp_rtn_ps_quandl_s

quandl_s =  ps_quandl_s.drop(ps_quandl_s[:start_time].index)
del ps_quandl_s

#----------------------------------------------------------------
#Resample the data taking the mean values for each month:

if use_quandl_returns == True:
	quandl_s = smp_rtn_quandl_s.copy()
if use_quandl_hist_vol == True:
	quandl_s = hv_smp_rtn_quandl_s.copy()

#================================================================
print "Processing gtrend data."

gtrend_df = pd.read_csv("gtrend_data.csv", sep=',')
gtrend_df = parse_csv_dates(gtrend_df, '%Y-%m-%d')

gtrend_df.fillna(method='ffill', inplace=True)
gtrend_df.fillna(method='bfill', inplace=True)

if use_gtrend_returns == True:
	gtrend_df.replace(0, np.nan, inplace=True); #gtrend_df.plot(); plt.show()
	smp_rtn_gtrend_df = gtrend_df.pct_change().fillna(0)#; smp_rtn_gtrend_df.plot(); plt.show()

	gtrend_df = smp_rtn_gtrend_df.copy()

#================================================================
print "Ensure quandl data matches gtrend index"

quandl_s = quandl_s.reindex(gtrend_df.index).fillna(method='ffill').fillna(method='bfill')

#================================================================

train_quandl_s = df_split(quandl_s, test_to_train_ratio)[0]
test_quandl_s = df_split(quandl_s, test_to_train_ratio)[1]

train_gtrend_df = df_split(gtrend_df, test_to_train_ratio)[0]
test_gtrend_df = df_split(gtrend_df, test_to_train_ratio)[1]

#================================================================

if feature_selection == True:
	print "Using descision tree for feature classification:"

	Y_train_np_array = np.asarray(train_quandl_s.as_matrix(), dtype="|S6")
	X_train_np_array = np.asarray(train_gtrend_df.as_matrix(), dtype="|S6")

	feature_classifier = RandomForestRegressor(n_estimators=estimators,random_state=0)
	feature_classifier = feature_classifier.fit(X_train_np_array, Y_train_np_array)

	feature_classifier_s = pd.Series(dict(zip(list(train_gtrend_df),feature_classifier.feature_importances_)))
	feature_classifier_s.sort_values(axis=0, ascending=False, inplace=True)

	feature_classifier_list = feature_classifier_s.head(feature_number).index.tolist()
	feature_classifier_s.to_csv("feature_classifier.csv")

	train_gtrend_df = train_gtrend_df[feature_classifier_list]
	test_gtrend_df = test_gtrend_df[feature_classifier_list]

#================================================================
if preprocess == True:
	
	low_var_threshold = low_var_mult*train_gtrend_df.var().max()
	print "Remove gtrend data with variance below the following: " + str(low_var_threshold)

	removed_low_var_index = train_gtrend_df.var().where(train_gtrend_df.var() > low_var_threshold).dropna().index
	train_gtrend_df = train_gtrend_df[removed_low_var_index]
	test_gtrend_df = test_gtrend_df[removed_low_var_index]

	#----------------------------------------------------------------

	f = lambda x: abs(x)
	crude_oil_price = crude_oil_type + "_price"

	train_corrs = train_gtrend_df.copy()
	train_corrs[crude_oil_price] = train_quandl_s
	train_corrs = train_corrs.corr().apply(f)
	train_corrs.sort_values(by=[crude_oil_price], inplace=True)

	train_correlations = train_gtrend_df.copy()
	train_correlations[crude_oil_price] = train_quandl_s
	train_correlations = train_correlations[train_corrs.index]; del train_corrs;
	train_correlations = train_correlations.corr().apply(f)

	high_correlation_threshold = high_corr_mult*train_correlations[crude_oil_price].mean()
	print "Remove gtrend data uncorrelated to the price: " + str(high_correlation_threshold)

	high_correlation_list= train_correlations.where(train_correlations[crude_oil_price] > high_correlation_threshold).dropna().index.tolist()
	high_train_correlations = train_correlations.where(train_correlations[crude_oil_price] > high_correlation_threshold).dropna()[high_correlation_list]
	high_train_correlations.to_csv("train_correlations.csv")
	high_correlation_list.remove(crude_oil_price)

	train_gtrend_df = train_gtrend_df[high_correlation_list]
	test_gtrend_df = test_gtrend_df[high_correlation_list]
	#print train_correlations.where(train_correlations[crude_oil_price] > high_correlation_threshold)

#================================================================

linear_regression_model = train_linear_regression(train_quandl_s, train_gtrend_df, lm_ply_deg)
regression_coeffs_np_array = linear_regression_model.coef_

keywords_used_s = pd.Series(list(train_gtrend_df)); print; print keywords_used_s
keywords_used_s.to_csv("keywords_used.csv", header=None, index=None)

#----------------------------------------------------------------

regression_model_df = pd.DataFrame()
regression_model_df["date"] = train_quandl_s.index
regression_model_df = regression_model_df.set_index('date')
regression_model_df["regression model"] = poly_transform(train_gtrend_df, lm_ply_deg).dot(regression_coeffs_np_array)
regression_model_df[crude_oil_type] = train_quandl_s

#----------------------------------------------------------------

results_df = fit_linear_regression(test_quandl_s, test_gtrend_df, linear_regression_model, lm_ply_deg)
results_df[crude_oil_type] = test_quandl_s

all_data_df = pd.DataFrame()
all_data_df["date"] = quandl_s.index
all_data_df = all_data_df.set_index("date")

all_data_df["regression model"] = regression_model_df["regression model"]
all_data_df["predicted model"] = results_df["predicted model"]
all_data_df[crude_oil_type] = quandl_s

del results_df; del regression_model_df

all_data_df.plot(); plt.show()

#================================================================

all_data_df["regression error"] = abs((all_data_df[crude_oil_type]-all_data_df["regression model"])/all_data_df[crude_oil_type])
all_data_df["predicted error"] = abs((all_data_df[crude_oil_type]-all_data_df["predicted model"])/all_data_df[crude_oil_type])

print;
print "regression error: " + str(all_data_df["regression error"].mean())
print "predicted error: " + str(all_data_df["predicted error"].mean())
