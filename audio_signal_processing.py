from __future__ import division
import tensorflow as tf
import numpy as np

class AudioSignalProcessiong():
	def __init__(self, audio = None):
		self.audio = audio

	def log10(x):
		num = tf.log(x)
		den = tf.log(tf.constant(10, dtype=num.dtype))
		return (tf.div(num, den))


	def overlapping_slicer_3D(_input, block_size, stride):
		_input_rank = int(len(_input.get_shape()))
		blocks = []
		n = _input.get_shape().as_list()[_input_rank - 1]
		low = range(0, n, stride)
		high = range(block_size, n + 1, stride)
		low_high = zip(low, high)
		for low, high in low_high:
			blocks.append(_input[:, low:high])
		return (tf.stack(blocks, _input_rank - 1))


	def angle(z):
		if z.dtype == tf.complex128:
			dtype = tf.float64
		elif z.dtype == tf.complex64:
			dtype = tf.float32
		else:
			raise ValueError('input z must be of type complex64 or complex128')

		x = tf.real(z)
		y = tf.imag(z)
		x_neg = tf.cast(x < 0.0, dtype)
		y_neg = tf.cast(y < 0.0, dtype)
		y_pos = tf.cast(y >= 0.0, dtype)
		offset = x_neg * (y_pos - y_neg) * np.pi
		return tf.atan(y / x) + offset


	def is_power2(x):
		return x > 0 and ((x & (x - 1)) == 0)


	def dft_analysis(_input, window, N):
		"""
		Analysis of a signal using the discrete Fourier transform inputs:
		_input: tensor of shape [batch_size, N] window: analysis window, tensor of shape [N]
		N: FFT size
		returns:
		Tensors m, p: magnitude and phase spectrum of _input
		m of shape [batch_size, num_coefficients]
		p of shape [batch_size, num_coefficients]
		"""

		if not(is_power2(N)):
			raise ValueError("FFT size is not a power of 2")

		_, input_length = _input.get_shape()
		_input_shape = tf.shape(_input)

		if (int(input_length) > N):
			raise ValueError("Input length is greater than FFT size")

		if (int(window.get_shape()[0]) != N):
			raise ValueError("Window length is different from FFT size")

		if int(input_length) < N:
			with tf.name_scope('DFT_Zero_padding'):
				zeros_left = tf.zeros(_input_shape)[:, :int((N - (int(input_length))+1) / 2)]
				zeros_right = tf.zeros(_input_shape)[:, :int((N - (int(input_length))) / 2)]
				_input = tf.concat([zeros_left, _input, zeros_right], axis=1)
				assert(int(_input.get_shape()[1]) == N)

		positive_spectrum_size = int(N/2) + 1
		with tf.name_scope('Windowing'):
			window_norm = tf.div(window, tf.reduce_sum(window))
			# window the input
			windowed_input = tf.multiply(_input, window_norm)

		with tf.name_scope('Zero_phase_padding'):
			# zero-phase window in fftbuffer
			fftbuffer_left  = tf.slice(windowed_input, [0, int(N/2)], [-1, -1])
			fftbuffer_right = tf.slice(windowed_input, [0, 0],   [-1, int(N/2)])
			fftbuffer = tf.concat([fftbuffer_left, fftbuffer_right], axis=1)
			fft = tf.spectral.rfft(fftbuffer)

		with tf.name_scope('Slice_positive_side'):
			sliced_fft = tf.slice(fft, [0, 0], [-1, positive_spectrum_size])

		with tf.name_scope('Magnitude'):
			# compute absolute value of positive side
			abs_fft = tf.abs(sliced_fft)

			# magnitude spectrum of positive frequencies in dB
			magnitude = 20 * log10(tf.maximum(abs_fft, 1E-06))

		with tf.name_scope('Phase'):
			# phase of positive frequencies
			phase = angle(sliced_fft)

		return magnitude, phase

	def stft_analysis(_input, window, N, H) :
		"""
		Analysis of a sound using the short-time Fourier transform
		Inputs:
		_input: tensor of shape [batch_size, audio_samples]
		window: analysis window, tensor of shape [N]
		N: FFT size, Integer
		H: hop size, Integer
		Returns:
		magnitudes, phases: 3D tensor with magnitude and phase spectra of shape
		[batch_size, coefficients, frames]
		"""
		if (H <= 0):
			raise ValueError("Hop size (H) smaller or equal to 0")
		if not(is_power2(N)):
			raise ValueError("FFT size is not a power of 2")

		_input_shape = tf.shape(_input)
		pad_size = int(N / 2)
		with tf.name_scope('STFT_Zero_padding'):
			zeros_left = tf.zeros(_input_shape)[:, :pad_size]
			zeros_right = tf.zeros(_input_shape)[:, :pad_size]
			_input = tf.concat([zeros_left, _input, zeros_right], axis=1)

		with tf.name_scope('overlapping_slicer'):
			sliced_input = overlapping_slicer_3D(_input, N, H)
		_, frames, _ = sliced_input.get_shape()

		with tf.name_scope('DFT_analysis'):
			reshaped_sliced_input = tf.reshape(sliced_input, (-1, N))
			m, p = dft_analysis(reshaped_sliced_input, window, N)

		with tf.name_scope('STFT_output_reshape'):
			magnitudes = tf.reshape(m, (-1, int(m.get_shape()[-1]), int(frames)))
			phases =     tf.reshape(p, (-1, int(p.get_shape()[-1]), int(frames)))

		return magnitudes, phases