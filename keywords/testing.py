import pandas as pd
import os

old_keywords_set = set(pd.read_csv("oldkeywords.txt", header=None).to_dict(orient='list')[0])
new_keywords_set = set(pd.read_csv("newkeywords.txt", header=None).to_dict(orient='list')[0])

#print len(old_keywords_set)

new_not_in_old_set = set()

#print len(new_keywords_set)

for keyword in new_keywords_set:
	if keyword not in old_keywords_set:
		new_not_in_old_set.add(keyword)

#print len(new_not_in_old_set)


output_file = open('output_data.txt', 'w')

for i in new_not_in_old_set:
	print i
	output_file.write(str(i) + "\n")
	
output_file.close()