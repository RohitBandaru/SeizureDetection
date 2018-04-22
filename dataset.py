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
	print(str(len(files))+" number of points")
	for i, file in enumerate(files):
		print(str(i+1)+"/"+str(len(files))+": extracting "+file)
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

	m_ictal = ictal_train.shape[0]
	m_non_ictal = ictal_train.shape[0]

	data = np.vstack([ictal_train, non_ictal_train])
	labels = np.hstack([np.ones((m_ictal)),np.zeros((m_non_ictal))])

	# shuffle data
	rand_idx = np.random.permutation(labels.shape[0])

	labels = labels[rand_idx]
	data  = data[rand_idx,:]

	return data, labels

def train_val_split(data,labels,split):
	m = data.shape[0]
	train_m = int(m*split)
	val_m = m - train_m

	train_data = data[:train_m,:]
	train_labels = labels[:train_m]

	val_data = data[train_m:,:]
	val_labels = labels[train_m:]

	return train_data, train_labels, val_data, val_labels


