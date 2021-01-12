#Import packages.
import pandas as pd
from pytrends.request import TrendReq
import os

#Create cluster data for a specific key word. In this case "Trump"
key_word = 'Trump'
os.chdir('C:/Users/12407/Desktop/Education/Projects/Google_Trends')
pytrend = TrendReq(geo='US')
pytrend.build_payload(kw_list=[key_word])
related_queries = pytrend.related_queries()
df_rq = list(related_queries.get(key_word).values())[0]
data = {'State_Name':[
	'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 
	'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 
	'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 
	'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 
	'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'
]}
data = pd.DataFrame(data) 
data = data.set_index('State_Name')
for ind in df_rq.index: 
	pytrend = TrendReq(geo='US')
	pytrend.build_payload(kw_list=[df_rq['query'][ind]])
	df = pytrend.interest_by_region(resolution='REGION', inc_low_vol=False, inc_geo_code=False)
	data = data.merge(df, left_index=True, right_index=True)
	print('loop index: ' + str(ind+1))
data.to_csv('cluster_data.csv')