import scipy.io as spio
from scipy import signal
import pywt
import matplotlib.pyplot as plt
import numpy as np

path1 = "data/patient_2/ictal train/patient_2_121.mat"

for i in range(1,2):
	path = "data/patient_"+str(i)+"/ictal train/patient_"+str(i)+"_1.mat"
	path_t = "data/patient_1/ictal train/patient_"+str(i)+"_1.mat"
	data = spio.loadmat(path)["data"]

	print(data.shape)
	print(np.var(data/(np.mean(data,axis=0)),axis=0))
	'''print(data.shape)
	wp = pywt.WaveletPacket2D(data=data, wavelet='db1', mode='symmetric')

	for i in range(0,wp.maxlevel+1):
		print(np.log(np.sum(np.abs(wp.get_level(i)[0].data.shape))))
	
	f, t, Sxx = signal.spectrogram(data[:,10], 5000)
	print(Sxx/np.max(Sxx))
	plt.pcolormesh(t, f, Sxx)
	plt.show()
	'''
	#for node in wp.get_level(4):
	#	print(node.data.shape)