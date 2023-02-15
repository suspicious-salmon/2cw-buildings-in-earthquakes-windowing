import numpy as np
import matplotlib.pyplot as plt

# DFT repeats slice to make it PERIODIC, then does fourier transform.
# for non-periodic functions, must pad with zeros.
# or alternatively (as done here) make window f with number of items n < N and use np.fft.fft(f, N) which does this for us.

# Hamming (vs Hanning) reduces amplitude of first sidelobe at the expense of slower decay for higher frequencies.

def fft(f):
    ret_fft = np.fft.fft(f, N) * k*np.pi/N
    return np.abs(ret_fft)[:N//2]

pulse_period = 1 # period of square pulse (including both -ve and +ve t!)
k = 2000
sample_period = k*(pulse_period)
N = 2000000 # number of sample points
tau = sample_period/N # sample spacing
f_s = 1/tau # sample frequency

# | TIME DOMAIN

k_p = 20
W_0 = 2 * np.pi / sample_period
w_0 = 2 * np.pi / pulse_period
sineshift_full = 1/3 * 2 * np.pi / (W_0 * k_p)
sineshift_pulse = 1/3 * 2 * np.pi / (w_0 * k_p)

t = np.linspace(-sample_period/2, sample_period/2, N, endpoint=False)
t_discontinuous = np.linspace(-sample_period/2, sample_period/2 + sineshift_full, N, endpoint=False)
t_pulse = np.linspace(-pulse_period/2, pulse_period/2, int(N*pulse_period/sample_period), endpoint=False)
t_pulse_discontinuous = np.linspace(-pulse_period/2, pulse_period/2 + sineshift_pulse, int(N*pulse_period/sample_period), endpoint=False)

# Unpadded (full sample width)

sine_continuous = np.sin(k_p*W_0*t)
sine_discontinuous = np.sin(k_p*W_0*t_discontinuous)

hanning_continuous = 0.5 * (1 + np.cos(W_0*t))
hanning_discontinuous = 0.5 * (1 + np.cos(2*np.pi / (sample_period + 2 * sineshift_full) * t_discontinuous))

sine_continuous_hanning = sine_continuous * hanning_continuous
sine_discontinuous_hannning = sine_discontinuous * hanning_discontinuous

# Padded (smaller pulse width)

padded_sine_continuous = np.sin(k_p*w_0*t_pulse)
padded_sine_discontinuous = np.sin(k_p*w_0*t_pulse_discontinuous)

padded_hanning_continuous = 0.5 * (1 + np.cos(w_0*t_pulse))
padded_hanning_discontinuous = 0.5 * (1 + np.cos(2 * np.pi / (pulse_period + 2 * sineshift_pulse) * t_pulse_discontinuous))

padded_sine_continuous_hanning = padded_sine_continuous * padded_hanning_continuous
padded_sine_discontinuous_hanning = padded_sine_discontinuous * padded_hanning_discontinuous

# hammingPulse = 0.54 + 0.46 * np.cos(t_pulse * 2 * np.pi/pulse_period)
# WINDOW_PULSES.append(hammingPulse)

# | FREQUENCY DOMAIN

F_sine_continuous = fft(sine_continuous)
F_sine_discontinuous = fft(sine_discontinuous)

F_padded_sine_continuous = fft(padded_sine_continuous)
F_padded_sine_discontinuous = fft(padded_sine_discontinuous)

freq = np.fft.fftfreq(N, tau)[:N//2]
freq2 = 1/(N*tau) * np.arange(0, N, 1)[:N//2] # equivalent to above function.

# | PLOTS

fig, ax = plt.subplots(3, 4)
plt.subplots_adjust(hspace=0.2)

# for i in range(len(ax)):
#     for j in range(len(ax[i])):
#         ax[i,j].grid()

F_xlim = 50

ax[0,0].set_title("No discontinuities at ends")
ax[0,1].set_title("Discontinuities at ends")
ax[0,2].set_title("No discontinuities at ends\nHamming windowed")
ax[0,3].set_title("Discontinuities at ends\nHamming windowed")

ax[0,0].set_ylabel(f"Sine wave covering {k_p} periods")
ax[1,0].set_ylabel("FFT, no padding")
ax[2,0].set_ylabel("FFT, large padding")

# Sines without any windowing (square window)

ax[0,0].plot(t_pulse, padded_sine_continuous)
ax[0,0].set_xlim(t_pulse[0], t_pulse[-1])
ax[0,0].set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
ax[0,1].plot(t_pulse_discontinuous, padded_sine_discontinuous)
ax[0,1].set_xlim(t_pulse_discontinuous[0], t_pulse_discontinuous[-1])
ax[0,1].set_xticks([-0.5, -0.25, 0, 0.25, 0.5])

ax[1,0].plot(freq*k, fft(sine_continuous))
ax[1,0].set_xlim(0,F_xlim)
ax[1,0].set_ylim(bottom=0)
ax[1,1].plot(freq*k, fft(sine_discontinuous))
ax[1,1].set_xlim(0,F_xlim)
ax[1,1].set_ylim(bottom=0)

ax[2,0].plot(freq, fft(padded_sine_continuous))
ax[2,0].set_xlim(0,F_xlim)
ax[2,0].set_ylim(bottom=0)
ax[2,1].plot(freq, fft(padded_sine_discontinuous))
ax[2,1].set_xlim(0,F_xlim)
ax[2,1].set_ylim(bottom=0)

# Sines with hanning windowing

ax[0,2].plot(t_pulse, padded_sine_continuous_hanning)
ax[0,2].set_xlim(t_pulse[0], t_pulse[-1])
ax[0,2].set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
ax[0,3].plot(t_pulse_discontinuous, padded_sine_discontinuous_hanning)
ax[0,3].set_xlim(t_pulse_discontinuous[0], t_pulse_discontinuous[-1])
ax[0,3].set_xticks([-0.5, -0.25, 0, 0.25, 0.5])

ax[1,2].plot(freq*k, fft(sine_continuous_hanning))
ax[1,2].set_xlim(0,F_xlim)
ax[1,2].set_ylim(bottom=0)
ax[1,3].plot(freq*k, fft(sine_discontinuous_hannning))
ax[1,3].set_xlim(0,F_xlim)
ax[1,3].set_ylim(bottom=0)

ax[2,2].plot(freq, fft(padded_sine_continuous_hanning))
ax[2,2].set_xlim(0,F_xlim)
ax[2,2].set_ylim(bottom=0)
ax[2,3].plot(freq, fft(padded_sine_discontinuous_hanning))
ax[2,3].set_xlim(0,F_xlim)
ax[2,3].set_ylim(bottom=0)

plt.show()