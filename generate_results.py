import feature_extractor as fe 
from os import listdir
import scipy.io as spio
import numpy as np
import dataset

for patient_number in range(1,8):
	channels = dataset.get_channels(patient_number, 3, 200)
	print(channels)
	data, labels = dataset.get_data(patient_number, channels)

	print(data.shape)
	print(labels.shape)
	data = np.nan_to_num(data)
	#std = np.std(data, axis=0)
	#mean = np.mean(data, axis=0)
	max1 = np.max(data, axis=0)
	data2 = data/max1
	#data = np.nan_to_num((data-mean)/std)
	'''
	from sklearn.decomposition import PCA
	pca = PCA(n_components=50, whiten = True)
	pca.fit(data)
	data2 = pca.transform(data)
	'''
	train_data, train_labels, val_data, val_labels = dataset.train_val_split(np.nan_to_num(data2.astype(np.float64)), labels, .75)

	print(train_data.shape)
	print(train_labels.shape)
	print(val_data.shape)
	print(val_labels.shape)

	from sklearn.svm import SVC
	from sklearn.model_selection import KFold
	from sklearn import metrics
	from sklearn.ensemble import RandomForestClassifier
	#### SVM
	#clf = SVC(C=10, gamma=1000, kernel='rbf',
	 #  max_iter=10000000, probability=True)
	clf = RandomForestClassifier(n_estimators=10, max_depth=1, max_features = 20, random_state=0)
	clf.fit(train_data, train_labels)

	print("training")
	print(metrics.classification_report(train_labels,clf.predict(train_data)))
	print(metrics.roc_auc_score(train_labels, clf.predict_proba(train_data)[:,1]))
	print("\nvalidation")
	print(metrics.classification_report(val_labels,clf.predict(val_data)))
	print(metrics.roc_auc_score(val_labels,clf.predict_proba(val_data)[:,1]))
	print(metrics.accuracy_score(val_labels,clf.predict(val_data)))
	print("\n")

	fpr, tpr, thresholds = metrics.roc_curve(val_labels, clf.predict_proba(val_data)[:,1], pos_label=1)
	import matplotlib.pyplot as plt
	plt.plot(fpr,tpr)
	print(thresholds)

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
	        vec = np.nan_to_num(vec/max1)
	        vec = np.nan_to_num(vec.astype(np.float32))
	        filename = file[:-4].replace("test_", "")
	        score = clf.predict_proba(vec.reshape(-1, 1).T)[0,1]
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

