import pandas as pd
import re
data=pd.read_csv('train.csv')
for x in range(0,len(data['text'])):
	data['text'][x]=str(data['text'][x]).lower()
	data['text'][x]=data['text'][x]+" "
	data['keyword'][x]=str(data['keyword'][x]).lower()
	data['text'][x]=re.sub('http(.)*[ ]',"",data['text'][x])
	data['text'][x]=re.sub('[^a-zA-Z0-9]',' ',data['text'][x])
	if data['text'][x].find(data['keyword'][x])==-1:
		data['text'][x]=data['text'][x]+data['keyword'][x]
data.to_csv('train_preprocessed.csv',index=False)

