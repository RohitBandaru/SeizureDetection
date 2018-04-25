import numpy as np
import pywt
from pywt import wavedec
import scipy.io as spio
from scipy import signal
import scipy.signal as sg
'''
data: (t,c)
'''
def extract_feature(data):
	n_t, n_c = data.shape

	channels_num = 3
        #print("xxx")
	#window = sg.get_window('hamming', numpy.asarray(len(data)) )
        #print(type(window))
	#print(window)
	#high_pass = sg.firwin(numtaps = 48, cutoff = 0.16, width = None, window = window )
        
	channels = np.argsort(-np.var(data, axis=0))[0:channels_num]
	data = data[:,channels]
	n_c = channels_num
	feature = np.zeros(n_c*4)
	for channel_number in range(n_c):
		#filtered_data = sg.filtfilt(high_pass, [1.0], numpy.asarray(data[:,channel_number]))
		#print(filtered_data)
		wp = pywt.WaveletPacket(data=data[:,channel_number], wavelet='db4', mode='symmetric') #data[:,channel_number]
		for i in range(4):
			n_l = len(wp.get_level(i+1))
			for j in range(n_l):
				feature[channel_number*4+i] += np.log(np.sum(wp.get_level(i+1)[j].data).clip(min=0.00000001))
	window_num = 1

	ll = line_length(data).flatten()
	en = energy(data).flatten()
	va = variance(data).flatten()
	po = power(data).flatten()

	feature = np.append(feature, [ll,en,va,po])
	return feature

def extract_feature2(data):

	n_t, n_c = data.shape

	channels_num = 3
        #print("xxx")
	#window = sg.get_window('hamming', numpy.asarray(len(data)) )
        #print(type(window))
	#print(window)
	#high_pass = sg.firwin(numtaps = 48, cutoff = 0.16, width = None, window = window )
	channels = np.argsort(-np.var(data, axis=0))[0:channels_num]
	data = data[:,channels]
	n_c = channels_num
	feature = []

	for channel_number in range(n_c):
		#filtered_data = sg.filtfilt(high_pass, [1.0], numpy.asarray(data[:,channel_number]))
		#print(filtered_data)
		coeffs = wavedec(data = data[:, channel_number], wavelet='db4', mode = 'symmetric', level = 5)
		cA5, cD5, cD4, cD3, cD2, cD1 = coeffs
		for c in coeffs:
			en = energy_coeff(c)
			va = variance_coeff(c)
			abs_co = abs_val_coeff(c)
			std = stdev_coeff(c)
		
			feature.extend([en, va, abs_co, std])

	feature = np.asarray(feature)
	return feature
	
# not axis function

def abs_val_coeff(data):
	return np.mean(np.abs(data))

def stdev_coeff(data):
	return np.std(data)

def energy_coeff(data):
	#m, n = data.shape
	energy = np.sum(np.square(np.abs(data)))
	return energy

def variance_coeff(data):
	variance = np.sum(np.square(data - np.mean(data)))
	return variance


# axis functions

def stdev(data):
	return np.std(data, axis = 0)

def line_length(data):
	#m, n = data.shape
	# subtract previous data point from every point to get the line length
	data[1:,:] = np.abs(data[1:,:] - data[0:-1,:])
	line_length  = np.sum(data, axis = 0)
	return line_length

def energy(data):
	#m, n = data.shape
	energy = np.sum(np.square(np.abs(data)), axis = 0)
	return energy

def variance(data):
	variance = np.sum(np.square(data - np.mean(data, axis = 0)), axis = 0)
	return variance

def power(data):
	m, n = data.shape
	power = np.zeros(n)
	for j in range(n):
		# f contains the frequencies that correspond to the powers in Pxx_den
		f, Pxx_den = signal.welch(data, m)
		# sum up the powers in the frequency bands
		band_f = np.where(np.logical_or(np.logical_and(f>=12, f<=30), np.logical_and(f>=100, f<=600)))
		# add the power to the channel
		power[j] = np.sum(Pxx_den[band_f])

	return power
