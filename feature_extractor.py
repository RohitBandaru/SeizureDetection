import numpy as np
import pywt

'''
data: (t,c)
'''
def extract_feature(data):
	n_t, n_c = data.shape
	feature = np.zeros(n_c*4)
	for channel_number in range(n_c):
		wp = pywt.WaveletPacket(data=data[:,channel_number], wavelet='db1', mode='symmetric')
		for i in range(4):
			n_l = len(wp.get_level(i+1))
			for j in range(n_l):
				feature[channel_number*4+i] += np.log(np.sum(wp.get_level(i+1)[j].data).clip(min=0.00000001))
	return feature
