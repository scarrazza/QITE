import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{siunitx}')
plt.rc('font', family='serif', size=10)
plt.rc('pdf', compression=9)
plt.rc('axes', labelsize='large')


parser = argparse.ArgumentParser()
parser.add_argument("filename", nargs='+', type=str)
args = vars(parser.parse_args())


def plot_band(axs, x, cv, std, label=None):
    axs.plot(x, cv, '.-',label=label)
    axs.fill_between(x, cv-std, cv+std, alpha=0.2)


def main(filename):
    """Main function for simulation.
    """
    odata = pd.DataFrame()
    for f in filename:
        odata = odata.append(pd.read_csv(f))

    nqubits_range = np.sort(odata['nqubits'].unique())
    cv = []
    for nqubits in nqubits_range:
        data = odata[odata['nqubits'] == nqubits]
        beta_range = data['beta'].unique()
        rmean = np.abs(np.array([ (1 - data[data['beta'] == b]['AA']/data[data['beta'] == b]['F_fit']).mean() for b in beta_range]))
        cv.append(beta_range[np.argmin(rmean)])

    title = str(odata["hamiltonian"].iloc[0]).replace('_', ' ') + ' - $\\beta_c(n)$'

    plt.title(title)
    plt.plot(nqubits_range, cv, 'o')
    plt.xlabel('nqubits')

    def exponential(x, a, b):
        return a*np.exp(b*x)
    def base2(x, a, b):
        return a * 2**(b*x)

    pars, cov = curve_fit(f=exponential, xdata=nqubits_range, ydata=cv)
    plt.plot(nqubits_range, exponential(nqubits_range, *pars), label=f'$\\beta_c(n) = {pars[0]:.2f} \cdot e^{{ {pars[1]:.2f} \cdot n}}$')

    pars, cov = curve_fit(f=base2, xdata=nqubits_range, ydata=cv)
    plt.plot(nqubits_range, base2(nqubits_range, *pars), '--', label=f'$\\beta_c(n) = {pars[0]:.2f} \cdot 2^{{ {pars[1]:.2f} \cdot n}}$')

    plt.legend(frameon=False)

    output = str(odata["hamiltonian"].iloc[0]) + '_betac.pdf'
    print(f'Saving {output}')
    plt.savefig(f"{output}", bbox_inches='tight')


if __name__ == "__main__":
    main(**args)
