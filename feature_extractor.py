import numpy as np
import pywt
import scipy.io as spio
from scipy import signal

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
	window_num = 1
	channels_num = 5

	ll = line_length(data, window_num, channels_num).flatten()
	en = energy(data, window_num, channels_num).flatten()
	va = variance(data, window_num, channels_num).flatten()
	po = power(data, window_num, channels_num).flatten()

	feature = np.append(feature, [ll,en,va,po])
	return feature



# HW2 Functions


def line_length(data, window_num, channels_num):
	m, n = data.shape
	# subtract previous data point from every point to get the line length
	data[1:,:] = np.abs(data[1:,:] - data[0:-1,:])
	line_length = None
	window_size = int(m/window_num)
	for i in range(int(window_size)):
		# select a one second window
		window = data[window_size*i:window_size*(i+1),:]
		# sum the lengths in a window
		window_ll = np.sum(window, axis = 0)
		# add this line length to the data
		if(line_length is None):
			line_length = window_ll
		else:
			line_length = np.vstack([line_length, window_ll])
	# sort channels in descending order by mean value, 
	# and select the 5 channels with the highest max line length
	channels = np.argsort(np.max(-line_length, axis = 0))[0:channels_num]
	print("most important channels: ", channels)
	line_length = line_length[:,channels]
	return line_length

def energy(data, window_num, channels_num):
	m, n = data.shape
	energy = None
	window_size = int(m/window_num)
	# select a one second window
	for i in range(int(window_size)):
		window = data[window_size*i:window_size*(i+1),:]
		# compute the energy for each channel within the window
		window_energy = np.sum(np.square(np.abs(window)), axis = 0)
		# add window data to the energy datqa
		if(energy is None):
			energy = window_energy
		else:
			energy = np.vstack([energy, window_energy])
	# select the channels with the highest max energy
	channels = np.argsort(np.max(-energy, axis = 0))[0:channels_num]
	print("most important channels: ", channels)
	energy = energy[:,channels]
	return energy

def variance(data, window_num, channels_num):
	m, n = data.shape
	variance = None
	window_size = int(m/window_num)
	for i in range(int(window_size)):
		# select a one second window
		window = data[window_size*i:window_size*(i+1),:]
		# compute the variance for each channel within the window
		window_variance = np.sum(np.square(window - np.mean(window, axis = 0)), axis = 0)
		# add window data to the variance datqa
		if(variance is None):
			variance = window_variance
		else:
			variance = np.vstack([variance, window_variance])
	# select the channels with the highest max variance
	channels = np.argsort(np.max(-variance, axis = 0))[0:channels_num]
	print("most important channels: ", channels)
	variance = variance[:,channels]
	return variance

def power(data, window_num, channels_num):
	m, n = data.shape
	power = None
	window_size = int(m/window_num)
	for i in range(int(window_size)):
		window_power = np.zeros(n)
		# loop through channels to compute spectral power
		for j in range(n):
			window = data[window_size*i:window_size*(i+1),j]
			# f contains the frequencies that correspond to the powers in Pxx_den
			f, Pxx_den = signal.welch(window, window_size)
			# sum up the powers in the frequency bands
			band_f = np.where(np.logical_or(np.logical_and(f>=12, f<=30), np.logical_and(f>=100, f<=600)))
			# add the power to the channel
			window_power[j] = np.sum(Pxx_den[band_f])
		if(power is None):
			power = window_power
		else:
			power = np.vstack([power, window_power])
	# select the channels with the highest max spectral power in the bands
	channels = np.argsort(np.max(-power, axis = 0))[0:channels_num]
	print("most important channels: ", channels)
	power = power[:,channels]
	return power
