# SGN-14007 Project 
# Topic 1: Separation of drums from music signals
# Rafia Bushra 268449

import numpy as np
import matplotlib.pyplot as plot
import scipy
from scipy.io import wavfile
from scipy import signal
from scipy.signal import stft, istft


# h_p_separator() function separates the Harmonic and Percussive components
# of a monoaural audio signal.
# Parameters:
# $kmax = maximum number of iterations
# $file_name = name of audio file to be analyzed
# Return values:
# $h = harmonic component signal
# $p = percussive component signal
# $fs = sampling frequency
def h_p_separator ( kmax, file_name ):
    ## Read the audio signal.
    # fs = sampling frequency
    # ft = audio signal data
    fs, ft = wavfile.read(file_name)

    ## Calculate the STFT of input signal ft.
    winlen = 1024  # Length of FFT used in the function.
    f, t, F = stft(ft, fs, nperseg=512, nfft=winlen)  # F = STFT of ft.

    ## Calculate the range-compressed version of the power spectrogram.
    gamma = 0.3  # Following the value of gamma set by the reference paper
    W = np.power(np.abs(F), (2 * gamma))

    ## Set the initial values for the Harmonic and Percussive components.
    H = 0.5 * W  # Harmonic component
    P = 0.5 * W  # Percussive component

    ## Populate H and P.
    # Create the co-vectors for the convolve function.
    tempH = np.array([0.25, -0.5, 0.25], ndmin=2)
    tempP = np.array([[0.25], [-0.5], [0.25]])
    alpha = 0.3
    for k in range(kmax - 1):
        # Calculate the update variables.
        sumH = signal.convolve2d(H, tempH, mode='same')
        sumP = signal.convolve2d(P, tempP, mode='same')
        delta = (alpha * sumH) - ((1 - alpha) * sumP)
        # Update H and P.
        H = np.minimum(np.maximum(H + delta, np.zeros(H.shape)), W)
        P = W - H

    ## Binarize the separation results.
    binH = W * (H < P)
    binP = W * (H >= P)

    ## Convert the binarized results into waveforms.
    _, h = istft((np.power(binH, 1 / (2 * gamma))) * np.exp(1j * np.angle(F)), fs, nperseg=512, nfft=winlen)
    _, p = istft((np.power(binP, 1 / (2 * gamma))) * np.exp(1j * np.angle(F)), fs, nperseg=512, nfft=winlen)

    ## Calculate SNR 
    deviation = ft - (h[0:ft.size] + p[0:ft.size])
    SNR = 10 * np.log10(np.sum(np.power(ft, 2)) / np.sum(np.power(deviation, 2)))
    print('Signal-to-noise ratio for k = {} is {}'.format(kmax, SNR))

    ## Plot the results.
    # Orginal signal
    originalLog = 20 * np.log10(np.abs(F) + 1e-3)
    plot.pcolormesh(t, f, originalLog)
    plot.title('STFT Spectrogram of Original Audio Signal')
    plot.ylabel('Frequency (Hz)')
    plot.xlabel('Time (sec)')
    plot.savefig('Original.png')
    # Harmonic component
    harmonicLog = 20 * np.log10(np.abs(binH) + 1e-3)
    plot.pcolormesh(t, f, harmonicLog)
    plot.title('STFT Spectrogram of Harmonic Component at k={}'.format(kmax))
    plot.ylabel('Frequency [Hz]')
    plot.xlabel('Time [sec]')
    plot.savefig('Harmonic_k{}.png'.format(kmax))
    # Percussive component
    percussiveLog = 20 * np.log10(np.abs(binP) + 1e-3)
    plot.pcolormesh(t, f, percussiveLog)
    plot.title('STFT Spectrogram of Percussive Component at k={}'.format(kmax))
    plot.ylabel('Frequency [Hz]')
    plot.xlabel('Time [sec]')
    plot.savefig('Percussive_k{}.png'.format(kmax))

    return h, p, fs

#-------------------------------------------------------------------------------------------

## Tests
kmax = [5, 10, 100]
test1 = 'police03short.wav'
for k in kmax:
    h, p, fs = h_p_separator(k, test1)
    scipy.io.wavfile.write('Harmonic_k{}.wav'.format(k), fs, np.int16(h))
    scipy.io.wavfile.write('Percussive_k{}.wav'.format(k), fs, np.int16(p))