import scipy.io as spio
from scipy import signal
import pywt
import matplotlib.pyplot as plt
import numpy as np

path1 = "data/patient_2/ictal train/patient_2_121.mat"

path = "data/patient_1/ictal train/patient_1_1.mat"
path_t = "data/patient_1/ictal train/patient_1_1.mat"
data = spio.loadmat(path1)["data"]

print(data.shape)
print(np.var(data/(np.mean(data,axis=0)),axis=0))
channels_num = 10
channels = np.argsort(-np.var(data, axis=0))[0:channels_num]
print(channels)
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