import feature_extractor as fe 
from os import listdir
import scipy.io as spio
import numpy as np
'''
data: (n,m)
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
			data = np.hstack([data, vec])
	return data


def get_data(patient_number):
	ictal_train = directory_data("data/patient_"+str(patient_number)+"/ictal train/")
	non_ictal_train = directory_data("data/patient_"+str(patient_number)+"/non-ictal train/")

	n_ictal = ictal_train.shape[1]
	n_non_ictal = ictal_train.shape[1]

	train_data = np.hstack([ictal_train, non_ictal_train])
	labels = np.vstack([np.ones((n_ictal,1)),np.zeros((n_non_ictal,1))])

	# shuffle data
	rand_idx = np.random.permutation(labels.shape[0])

	labels = labels[rand_idx]
	train_data  = train_data[:,rand_idx]

	return train_data, labels



'''
data, labels = get_data(1)

print(data.shape)
print(labels)
print(labels.shape)
'''