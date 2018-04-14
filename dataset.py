import feature_extractor as fe 
from os import listdir
import scipy.io as spio
import numpy as np
'''
data: (m,n)
'''
def directory_data(path):
	data = None
	files = listdir(path)
	for i, file in enumerate(files):
		file_data = spio.loadmat(path+file)["data"]
		vec = fe.extract_feature(file_data)
		if data is None:
			data = vec
		else:
			data = np.vstack([data, vec])
	return data


def get_data(patient_number):
	ictal_train = directory_data("data/patient_"+str(patient_number)+"/ictal train/")
	non_ictal_train = directory_data("data/patient_"+str(patient_number)+"/non-ictal train/")

	n_ictal = ictal_train.shape[1]
	n_non_ictal = ictal_train.shape[1]

	train_data = np.vstack([ictal_train, non_ictal_train])
	labels = np.hstack([np.ones((n_ictal)),np.zeros((n_non_ictal))])

	# shuffle data
	rand_idx = np.random.permutation(labels.shape[0])

	labels = labels[rand_idx]
	train_data  = train_data[rand_idx,:]

	return train_data, labels

def train_val_split(data,labels,split):
	m = data.shape[0]
	train_m = int(m*split)
	vali_m = m - train_m

	train_data = data[:train_m,:]
	train_labels = labels[:train_m]

	val_data = data[train_m:,:]
	val_labels = labels[train_m:]

	return train_data, train_labels, val_data, val_labels

data, labels = get_data(1)

train_data, train_labels, val_data, val_labels = train_val_split(data, labels, .7)

from sklearn.svm import SVC
clf = SVC()
clf.fit(train_data, train_labels)
y_pred = clf.predict(val_data)

from sklearn.metrics import accuracy_score
print(y_pred)
print(accuracy_score(y_pred, val_labels))


