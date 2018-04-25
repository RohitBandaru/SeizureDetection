import scipy.io as spio
from scipy import signal
import pywt
import matplotlib.pyplot as plt
import numpy as np
import feature_extractor as fe
path1 = "data/patient_2/ictal train/patient_2_121.mat"

path = "data/patient_1/ictal train/patient_1_1.mat"
path_t = "data/patient_3/test/patient_3_test_1947.mat"
data = spio.loadmat(path_t)["data"]

print(data.shape)

vec = fe.extract_feature2(data)
print(vec)
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