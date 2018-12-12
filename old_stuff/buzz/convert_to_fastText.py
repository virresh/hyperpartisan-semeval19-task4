import pandas as pd

labels = []
text = []

with open('test_data.txt', 'r') as data_file:
    for line in data_file:
        article_text = line[7:]
        label = int(line[5])
        article_id = line[:4]
        labels.append(label)
        text.append(article_text.replace('\n',' '))

for x in range(0, len(labels)):
	print('__label__'+str(labels[x]), text[x])
