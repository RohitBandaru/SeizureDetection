import feature_extractor as fe 
from os import listdir
import scipy.io as spio
import numpy as np
import dataset

for patient_number in range(1,8):
	channels = dataset.get_channels(patient_number, 5, 150)
	print(channels)
	data, labels = dataset.get_data(patient_number, channels)
	datac, labelsc = dataset.get_data_coeff(patient_number, channels)

	data = np.nan_to_num(data)
	std = np.std(data, axis=0)
	mean = np.mean(data, axis=0)
	#max1 = np.max(data, axis=0)
	#data2 = data/max1
	data2 = np.nan_to_num((data-mean)/std)

	from sklearn.decomposition import FastICA
	from sklearn.decomposition import PCA
	pca = PCA(n_components=100, whiten = True)
	pca.fit(datac)
	data2c = pca.transform(datac)

	new_data = np.hstack([data, data2c])


	train_data, train_labels, val_data, val_labels = dataset.train_val_split(np.nan_to_num(new_data.astype(np.float64)), labels, .75)


	from sklearn.svm import SVC
	from sklearn.model_selection import KFold
	from sklearn import metrics
	from sklearn.ensemble import RandomForestClassifier
	#### SVM
	from sklearn.neural_network import MLPClassifier
	#### SVM
	#clf = SVC(C=100, gamma=1, kernel='rbf',
	 #max_iter=10000000, probability=True)
	clf = RandomForestClassifier(n_estimators=100, max_depth=10, max_features=5)
	clf.fit(train_data, train_labels)

	path = "data/patient_"+str(patient_number)+"/test/"
	files = listdir(path)
	print(str(len(files))+" number of points")

	import os
	import csv
	csv_path = 'patient_roc' + str(patient_number)+'.csv'
	f = open(csv_path, "w")
	f.truncate()
	f.close()
	with open(csv_path, 'a') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		for i, file in enumerate(files):
			if file == ".DS_Store":
				continue
			print(str(i+1)+"/"+str(len(files))+": extracting "+file)
			file_data = spio.loadmat(path+file)["data"]
			vec = np.nan_to_num(fe.extract_feature2(file_data[:,channels]))
			vec = np.nan_to_num((vec-mean)/std)

			vec2 = np.nan_to_num(fe.extract_feature_coeff(file_data[:,channels]))
			vec2 = np.nan_to_num(pca.transform(vec2.reshape(1,vec2.shape[0])).astype(np.float32))

			vec = np.append(vec,vec2)
			vec = vec.reshape(1,vec.shape[0])
			filename = file[:-4].replace("test_", "")
			score = clf.predict_proba(vec).reshape(-1, 1).T[0,1]
			pred = [filename, score]
			print(pred)

			spamwriter.writerow(pred)

###concatenate all files
import pandas as pd
files = ["patient_roc1.csv","patient_roc2.csv","patient_roc3.csv",
         "patient_roc4.csv","patient_roc5.csv","patient_roc6.csv","patient_roc7.csv"]

out = open("patient.csv", "w")
out.truncate()
out.close()

with open("patient.csv", 'a') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["id","prediction"])
    for file in files:
        with open(file, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                writer.writerow(row)
