import numpy as np
import matplotlib.pyplot as plt

# DFT repeats slice to make it PERIODIC, then does fourier transform.
# for non-periodic functions, must pad with zeros.
# or alternatively (as done here) make window f with number of items n < N and use np.fft.fft(f, N) which does this for us.

# Hamming (vs Hanning) reduces amplitude of first sidelobe at the expense of slower decay for higher frequencies.

def fft(f):
    return np.fft.fft(f, N) * k*np.pi/N

pulse_period = 1 # period of square pulse (including both -ve and +ve t!)
k = 2000
sample_period = k*(pulse_period)
N = 2000000 # number of sample points
tau = sample_period/N # sample spacing
f_s = 1/tau # sample frequency

# | TIME DOMAIN

WINDOW_NAMES = ["Square", "Hamming", "Hanning", "Blackman", "Quadratic"]
WINDOW_PULSES = []

t = np.linspace(-sample_period/2, sample_period/2, N, endpoint=False)
pulse_ones = np.ones(int(N*pulse_period/sample_period))
pulse_left = int(N/2+N*-pulse_period/sample_period/2)
pulse_right = int(N/2+N*pulse_period/sample_period/2)
t_pulse = t[pulse_left:pulse_right]

squarePulse = pulse_ones
WINDOW_PULSES.append(squarePulse)

hammingPulse = 0.54 + 0.46 * np.cos(t_pulse * 2 * np.pi/pulse_period)
WINDOW_PULSES.append(hammingPulse)

hanningPulse = 0.5 * (1 + np.cos(t_pulse * 2 * np.pi/pulse_period))
WINDOW_PULSES.append(hanningPulse)

blackmanPulse = 0.42 + 0.5 * np.cos(t_pulse * 2 * np.pi/pulse_period) + 0.08 * np.cos(2* t_pulse * 2 * np.pi/pulse_period)
WINDOW_PULSES.append(blackmanPulse)

quadraticPulse = pulse_ones*(1 - 4*t_pulse**2)
WINDOW_PULSES.append(quadraticPulse)

# | FREQUENCY DOMAIN

TRANSFORMS = [fft(f) for f in WINDOW_PULSES]

freq = np.fft.fftfreq(N, tau)[:N//2]
freq2 = 1/(N*tau) * np.arange(0, N, 1)[:N//2] # equivalent to above function.

# | PLOTS

fig, ax = plt.subplots(5, 3, sharex="col")
plt.subplots_adjust(hspace=0.2)

ax[4,0].set_xlabel("$\\frac{-T}{2} \\leq t \\leq \\frac{T}{2}$")
ax[4,1].set_xlabel("Frequency $f$")
ax[4,2].set_xlabel("Frequency $log(f)$")

ax[0,0].set_title("Window function f(t)")
ax[0,1].set_title("(dB) Fourier transform F(f), zoomed in")
ax[0,2].set_title("(dB) Zoomed out, log frequency scale")

for row in range(5):
    # Time domain representatio
    ax[row,0].plot(t_pulse, WINDOW_PULSES[row], label=WINDOW_NAMES[row])
    ax[row,0].legend()
    ax[row,0].set_yticks([0,0.25,0.5,0.75,1])
    ax[row,0].set_ylim(0, 1.1)
    ax[row,0].set_xlim(t_pulse[0], t_pulse[-1])
    ax[row,0].grid()

    # Frequency domain representation zoomed in
    ax[row,1].plot(freq, 20*np.log10(np.abs(TRANSFORMS[row][:N//2])))
    ax[row,1].set_ylim(-100,20)
    ax[row,1].set_xlim(0, 10)
    ax[row,1].grid()

    # Frequency domain representation zoomed out
    ax[row,2].plot(freq, 20*np.log10(np.abs(TRANSFORMS[row][:N//2])))
    ax[row,2].set_xscale('log')
    ax[row,2].set_ylim(-200,20)
    ax[row,2].set_xlim(1,250)
    ax[row,2].grid()

plt.show()