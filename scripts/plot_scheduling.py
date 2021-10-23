import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{siunitx}')
plt.rc('font', family='serif', size=10)
plt.rc('pdf', compression=9)
plt.rc('axes', labelsize='large')



def plot_band(axs, x, cv, std, label=None):
    axs.plot(x, cv, '-',label=label)
    axs.fill_between(x, cv-std, cv+std, alpha=0.1)


def do_scheduling(filename):
    data = pd.read_csv(filename)
    data = data.sort_values(by=['beta'])
    title = str(data["hamiltonian"].iloc[0]).replace('_', ' ') + ' hamiltonian' \
        ' - ' + str(data["nqubits"].iloc[0]) + ' qubits'
    beta_range = data['beta'].unique()
    gdata = data.groupby(['beta'])
    output = filename.replace('.csv', '').split('/')[-1]

    def schedule(t, params):
        return t**params

    for beta in beta_range:
        param = np.array([ float(i[1:-1]) for i in gdata.get_group(beta)["F_fit_params"].tolist() ])
        r = gdata.get_group(beta)["F_fit_r"].to_numpy()
        deltas = []
        for p, ir in zip(param, r):
            betas = np.array([ beta * schedule(step/ir, p) for step in range(1, ir+1)])
            deltas.append(betas[0])
        plt.figure(figsize=(5,1.7))
        plt.title(title)
        plt.hist(deltas, bins=50)
        plt.axvline(1e-3, color='k', label='$\\epsilon$') # epsilon
        plt.axvline(np.mean(deltas), color='r', label='mean first step')
        plt.xlabel(f"$\\beta$={beta}")
        plt.legend(frameon=False)
        out = output + f'_beta_{beta}.pdf'
        print(f'Saving {out}')
        plt.savefig(f"{out}", bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    do_scheduling('results/2002/weighted_maxcut/weighted_maxcut_qubits_12_maxbeta_2002_step_25_trials_1000.csv')
