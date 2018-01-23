import numpy as np
import quandl
import pandas as pd
import datetime as dt
from datetime import timedelta
from datetime import datetime

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge 

#================================================================

def datetime_range(start=None, end=None):
	span = end - start
	for i in xrange(span.days + 1):
		yield start + timedelta(days=i)

def dates_as_string(start=None, end=None):
	return start.strftime('%Y-%m-%d') + ' ' + end.strftime('%Y-%m-%d')

def parse_csv_dates(input_df=None, date_format=None):
	#This is to cleanup the .csv input files. The string '%Y-%m' is an example of the date format required
	
	output_df = input_df
	dt_strptime = lambda input_date: datetime.strptime(input_date, date_format)

	output_df['date'] = input_df['date'].map(dt_strptime)
	output_df = output_df.set_index('date')
	
	return output_df
	
def df_split(input_df=None, ratio=None):
	
	number_of_data_rows = len(input_df)

	top_half_no = int((number_of_data_rows)*ratio)
	bottom_half_no = number_of_data_rows - top_half_no

	top_half_df = input_df.head(top_half_no)
	bottom__df = input_df.tail(bottom_half_no)

	return top_half_df, bottom__df
	
#================================================================
#Linear Regression (polynomial features) functions:

def poly_transform(input_df=None, poly_deg=None):

	input_np_array = input_df.as_matrix()
	poly = PolynomialFeatures(degree=poly_deg)
	
	return poly.fit_transform(input_np_array)

def train_linear_regression(train_obs=None, train_exp=None, poly_degree=None):

	Y_train_np_array = train_obs.as_matrix()
	X_train_poly_np_array = poly_transform(train_exp, poly_degree)

	regression_model = LinearRegression(fit_intercept=False)
	regression_model.fit(X_train_poly_np_array, Y_train_np_array)

	return regression_model

def fit_linear_regression(test_obs=None, test_exp=None, regression_model=None, poly_degree=None):
	
	Y_test_np_array = test_obs.as_matrix()
	X_test_poly_np_array = poly_transform(test_exp, poly_degree)

	#Put all data in a new dataframe:
	results_df = pd.DataFrame()
	results_df["date"] = test_obs.index
	results_df = results_df.set_index("date")

	results_df["predicted model"] = regression_model.predict(X_test_poly_np_array)
	
	return results_df

#================================================================
#WIP
#================================================================

def train_bayesian_ridge(train_obs=None, train_exp=None, poly_degree=None):

	Y_train_np_array = train_obs.as_matrix()
	X_train_poly_np_array = poly_transform(train_exp, poly_degree)

	regression_model = Ridge(alpha = 0.1)
	regression_model.fit(X_train_poly_np_array, Y_train_np_array)

	return regression_model

	
	