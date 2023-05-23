'''
import pandas as pd
           
data = {}            
for part in ['TF_Coil','Vessel','Blanket']:
    for loop in ['in','out']:
        for comp in ['r','z']:
            data[(part,loop,comp)] = demo.parts[part][loop][comp]


df = pd.DataFrame(data)
             

print(df)


filename = './referance/DEMO1.csv'

df.to_csv(filename, sep=',')



import csv

test_array = []
test_array.append({'fruit': 'apple', 'quantity': 5, 'color': 'red'});
test_array.append({'fruit': 'pear', 'quantity': 8, 'color': 'green'});
test_array.append({'fruit': 'banana', 'quantity': 3, 'color': 'yellow'});
test_array.append({'fruit': 'orange', 'quantity': 11, 'color': 'orange'});
fieldnames = ['fruit', 'quantity', 'color']
test_file = open('test2.csv','wb')
csvwriter = csv.DictWriter(test_file, delimiter=',', fieldnames=fieldnames)
#csvwriter.writerow(dict((fn,fn) for fn in fieldnames))
for row in test_array:
     csvwriter.writerow(row)
test_file.close()
'''
import csv
with open('eggs.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(demo.parts.keys())
    writer.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])
