import csv
datas = csv.reader(open('predict_data_train.csv','r'))
for data in datas:
    print(data)
