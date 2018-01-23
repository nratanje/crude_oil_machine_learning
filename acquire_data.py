import numpy as np
import quandl
import pandas as pd

import datetime as dt
from datetime import timedelta
import time

from authentication_keys import quandl_api_key
from pytrends.request import TrendReq

from utility_functions import *

#================================================================

def download_quandl_data(start_time=None, end_time=None):

	OPEC_df = quandl.get("OPEC/ORB", trim_start = start_time, trim_end = end_time, authtoken = quandl_api_key)
	BRENT_df = quandl.get("EIA/PET_RBRTE_D", trim_start = start_time, trim_end = end_time, authtoken = quandl_api_key)
	WTI_df = quandl.get("EIA/PET_RWTC_D", trim_start = start_time, trim_end = end_time, authtoken = quandl_api_key)

	OPEC_df.rename(columns={'Value': 'OPEC'}, inplace=True)
	BRENT_df.rename(columns={'Value': 'BRENT'}, inplace=True)
	WTI_df.rename(columns={'Value': 'WTI'}, inplace=True)

	crude_oil_df = pd.concat([OPEC_df, BRENT_df, WTI_df], axis=1)
	crude_oil_df.to_csv('crude_oil.csv')
		
	print "quandl data downloaded to crude_oil.csv"

#================================================================

def download_gtrend_data(start_time=None, end_time=None, searchwords_list=None):
	
	pytrend = TrendReq()
	gtrend_df = pd.DataFrame()
	
	for searchword in searchwords_list:

		pytrend.build_payload(kw_list=[searchword], timeframe=dates_as_string(start_time,end_time))
		gtrend_searchword_df = pytrend.interest_over_time()
		
		try:
			gtrend_df = pd.concat([gtrend_df, gtrend_searchword_df[searchword]], axis=1)		
			print str(searchword) + " " + str(dates_as_string(start_time,end_time)); time.sleep(1)
			
		except:
			print "error with " + searchword; continue

	gtrend_df.to_csv('gtrend_data.csv')
	print; print "gtrend data downloaded to gtrend_data.csv"
