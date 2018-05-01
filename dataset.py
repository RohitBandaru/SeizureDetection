import feature_extractor as fe 
from os import listdir
import scipy.io as spio
import numpy as np


'''
data: (m,n)
'''
def directory_data(path, channels):
	files = listdir(path)
	print(str(len(files))+" number of points")
	data = None 

	for i, file in enumerate(files):
		print(str(i+1)+"/"+str(len(files))+": extracting "+file)
		file_data = spio.loadmat(path+file)["data"]
		vec = fe.extract_feature2(file_data[:,channels])
		if data is None:
			data = np.zeros((len(files),vec.shape[0]))
		data[i,:] = vec
	print("directory data shape", data.shape)
	return data


def get_data(patient_number, channels):
        #dir_name = "C:/Users/Aasta/Documents/CU SP 18/ECE 5040/"
	ictal_train = directory_data("data/patient_"+str(patient_number)+"/ictal train/", channels)
	non_ictal_train = directory_data("data/patient_"+str(patient_number)+"/non-ictal train/", channels)

	m_ictal = ictal_train.shape[0]
	m_non_ictal = non_ictal_train.shape[0]
	data = data = np.vstack([ictal_train, non_ictal_train])
	labels = np.hstack([np.ones((m_ictal)),np.zeros((m_non_ictal))])
	for i in range(int(m_non_ictal/m_ictal) -1):
		data = np.vstack([ictal_train, data])
		labels = np.hstack([np.ones((m_ictal)),labels])
	return data, labels

def directory_data_coeff(path, channels):
	files = listdir(path)
	print(str(len(files))+" number of points")
	data = None 

	for i, file in enumerate(files):
		print(str(i+1)+"/"+str(len(files))+": extracting "+file)
		file_data = spio.loadmat(path+file)["data"]
		vec = fe.extract_feature_coeff(file_data[:,channels])
		if data is None:
			data = np.zeros((len(files),vec.shape[0]))
		data[i,:] = vec
	print("directory data shape", data.shape)
	return data

def get_data_coeff(patient_number, channels):
        #dir_name = "C:/Users/Aasta/Documents/CU SP 18/ECE 5040/"
	ictal_train = directory_data_coeff("data/patient_"+str(patient_number)+"/ictal train/", channels)
	non_ictal_train = directory_data_coeff("data/patient_"+str(patient_number)+"/non-ictal train/", channels)

	m_ictal = ictal_train.shape[0]
	m_non_ictal = non_ictal_train.shape[0]
	data = np.vstack([ictal_train, non_ictal_train])
	labels = np.hstack([np.ones((m_ictal)),np.zeros((m_non_ictal))])
	for i in range(int(m_non_ictal/m_ictal) -1):
		data = np.vstack([ictal_train, data])
		labels = np.hstack([np.ones((m_ictal)),labels])
	return data, labels



def directory_data_sample(path,channels_num, sample):
	import random
	files = listdir(path)
	random.shuffle(files)
	data = None 

	for i, file in enumerate(files):
		if(i>sample):
			break
		file_data = spio.loadmat(path+file)["data"]
		vec = np.argsort(-np.var(file_data, axis=0))[0:channels_num]
		if data is None:
			data = np.zeros((len(files),vec.shape[0]))
		data[i,:] = vec
	return data

def get_channels(patient_number, channels_num, sample):
	ictal_train = directory_data_sample("data/patient_"+str(patient_number)+"/ictal train/", channels_num, sample)
	non_ictal_train = directory_data_sample("data/patient_"+str(patient_number)+"/non-ictal train/", channels_num, sample)
	data = np.vstack([ictal_train, non_ictal_train])
	return np.argsort(np.bincount(data.astype(int).flatten()))[-channels_num:]
	

def train_val_split(data,labels,split):
	# shuffle data
	rand_idx = np.random.permutation(labels.shape[0])

	labels = labels[rand_idx]
	data  = data[rand_idx,:]

	m = data.shape[0]
	train_m = int(m*split)
	val_m = m - train_m

	train_data = data[:train_m,:]
	train_labels = labels[:train_m]

	val_data = data[train_m:,:]
	val_labels = labels[train_m:]

	return train_data, train_labels, val_data, val_labels


