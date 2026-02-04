'''Create a plot for output and plot for FFT'''


import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.signal import welch
ROOT = Path(__file__).resolve().parents[2]
def get_esd(t, v, target_df=None, min_nperseg=256):

    # remove the mean/DC
    v = v - np.mean(v)

    #Calculate the sampling rate needed
    dt = np.mean(np.diff(t))
    fs = 1 / dt
    print(f"Sample rate: {fs:.1f} Hz")
    print(f"Duration: {t[-1] - t[0]:.2f} s")
    
    # Need to find nperseg good value for all of them
    # Choose nperseg
    if target_df is None:
        nperseg = len(v) // 2
    else:
        nperseg = int(fs / target_df)

   # Clamp nperseg to sensible limits
    nperseg = max(min_nperseg, min(nperseg, len(v)//2))

    # Welch PSD
    f, psd = welch(
        v,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg//2,
        scaling="density"
    )
    noise = 20 * np.log10(np.sqrt(psd))

    return [f, noise]

def get_paths(extensions):
    paths = []
    for ext in extensions:
        paths.append(ROOT / "data" / "raw" / "csv_data" / f"{ext}.csv")
    return paths

def get_data(paths):
    
    '''
    EXPECTED FORMAT FROM OSCILLOSCOPE
    X   | CH1
    (S) | (VOLT)
    ----------
    DATA|DATA
    '''
    data_arr = []
    for path in paths:
        data = np.genfromtxt(
            path,
            delimiter=",",
            skip_header=2,
            usecols=(0,1)
        )
        data = data[~np.isnan(data).any(axis=1)]
        t = data[:, 0]
        v = data[:, 1]
        print(f"For path {path}")
        print(f"Time in s and No. Samples{len(t)}")
        print(f"voltage in V and No. Samples{len(v)}")
        data_arr.append([t, v])
    return data_arr

# Fetch the CSV data
extensions = ["through_200ms", "through_2ms", "through_200us", "through_20us"]
paths = get_paths(extensions)



# Next I want to extract the x and y values
data = get_data(paths)
target_df =  [0.5, 50.0, 5000.0, 5000.0]
# Now get the x and y data for my diagram
plot_values = []
for i, d in enumerate(data):
    plot_values.append(get_esd(d[0], d[1], target_df=target_df[i]))

# --- Plot ---
plt.figure()


for i, val in enumerate(plot_values):

    plt.semilogx(val[0], val[1], label=f"{extensions[i]}")

plt.xlim(1, 1e5)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Error spectral density [dBV/√Hz]")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()
