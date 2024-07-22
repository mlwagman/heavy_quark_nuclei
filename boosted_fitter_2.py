import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re

# Function to read the CSV files and extract the data
def read_data(directory):
    # Updated pattern to match the filenames
    pattern = os.path.join(directory, "Hammys_LO_dtau0.3_Nstep300_Nwalkers1000_Ncoord2_Nc3_nskip10_Nf4_alpha0.75_spoila1.0_log_mu_r0.0_wavefunction_compact_potential_full_L*_afac1_masses*_mtm*.csv")
    files = glob.glob(pattern)

    print(f"Found {len(files)} files matching the pattern.")

    data = []
    for file in files:
        try:
            # Extract L from the filename using regex
            match = re.search(r"_L([0-9.]+)_", file)
            if match:
                L = float(match.group(1))
                df = pd.read_csv(file)
                mean = df['mean'].values[0]
                err = df['err'].values[0]
                data.append((L, mean, err))
                print(f"Successfully read data from {file}: L={L}, mean={mean}, err={err}")
            else:
                print(f"Failed to extract L from {file}")
        except Exception as e:
            print(f"Failed to read data from {file}: {e}")
    
    return data

# Linear function for fitting
def linear(x, a, b):
    return a * x + b

# Main function to read data, fit and plot
def main():
    directory = 'mtm_data'
    data = read_data(directory)
    
    if not data:
        print("No data to process. Exiting.")
        return
    
    # Sort data by L for better plotting
    data.sort(key=lambda x: x[0])
    
    Ls, means, errs = zip(*data)
    
    # Fit the data to a linear model
    popt, pcov = curve_fit(linear, Ls, means, sigma=errs)
    a, b = popt
    fit_line = linear(np.array(Ls), a, b)
    
    # Plot the data with error bars
    plt.errorbar(Ls, means, yerr=errs, fmt='o', label='Data')
    plt.plot(Ls, fit_line, label=f'Fit: y = {a:.2e}x + {b:.2e}', color='red')
    plt.xlabel('L')
    plt.ylabel('mean')
    plt.title('L vs mean with linear fit')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
