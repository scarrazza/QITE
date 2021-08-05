from numpy.random import beta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage{siunitx}')
#plt.rc('font', family='serif', size=10)
# plt.rc('pdf', compression=9)
plt.rc('axes', labelsize='large')
#plt.rc('title', labelsize='large')


def plot_band(axs, x, cv, std, label=None):
    axs.plot(x, cv, '-',label=label)
    axs.fill_between(x, cv-std, cv+std, alpha=0.1)


def do_beta(filename, show_ylabel=True):
    data = pd.read_csv(filename)
    data = data.sort_values(by=['beta'])
    title = str(data["hamiltonian"].iloc[0]).replace('_', ' ') + ' hamiltonian' \
        ' - $N=$' + str(data["nqubits"].iloc[0])
    if data["hamiltonian"].iloc[0] == 'weighted_maxcut':
        title = 'Weighted MaxCut'
    elif data["hamiltonian"].iloc[0] == 'rbm':
        title = 'Quantum RBMs'
    elif data["hamiltonian"].iloc[0] == 'heisenberg':
        title = 'Quantum spin glasses'
    elif str(data["hamiltonian"].iloc[0]) == 'maxcut':
        title = 'MaxCut'

    beta_range = data['beta'].unique()
    gdata = data.groupby(['beta'])
    means = gdata.mean()
    stds = gdata.std()

    fig, axs = plt.subplots(2, 1, figsize=(4,5), sharex=True)
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    axs[0].set_title(title)
    plot_band(axs[0], beta_range, means['F'], stds['F'], label='F.U.')
    plot_band(axs[0], beta_range, means['F_fit'], stds['F_fit'], label='F.NU.')
    plot_band(axs[0], beta_range, means['AA'], stds['AA'], label='C.')
    plot_band(axs[0], beta_range, means['C'], stds['C'], label='P.')

    axs[0].set_yscale('log')
    axs[0].legend(frameon=False, ncol=2)
    if show_ylabel:
        axs[0].set_ylabel('Query complexity')
        axs[1].set_ylabel('Query depth');
    axs[1].set_yscale('log')

    plot_band(axs[1], beta_range, means['F_depth'], stds['F_depth'], label='F.U.')
    plot_band(axs[1], beta_range, means['F_fit_depth'], stds['F_fit_depth'], label='F.NU.')
    plot_band(axs[1], beta_range, means['AA'], stds['AA'], label='C.')
    plot_band(axs[1], beta_range, means['C_depth'], stds['C_depth'], label='P.')

    axs[1].legend(frameon=False, ncol=2)
    axs[1].set_xlabel(r'$\beta$')

    output = filename.replace('.csv', '.pdf').split('/')[-1]
    print(f'Saving {output}')
    plt.savefig(f"{output}", bbox_inches='tight')
    plt.close()

def do_critical(filename, skip=False, show_ylabel=True):
    odata = pd.DataFrame()
    for f in filename:
        odata = odata.append(pd.read_csv(f))

    nqubits_range = np.sort(odata['nqubits'].unique())
    if skip:
        nqubits_range = nqubits_range[:-1]
    cv = []
    for nqubits in nqubits_range:
        data = odata[odata['nqubits'] == nqubits]
        beta_range = data['beta'].unique()
        rmean = np.abs(np.array([ (1 - data[data['beta'] == b]['AA']/data[data['beta'] == b]['F_fit']).mean() for b in beta_range]))
        cv.append(beta_range[np.argmin(rmean)])

    title = str(odata["hamiltonian"].iloc[0]).replace('_', ' ') + ' hamiltonian'
    if str(odata["hamiltonian"].iloc[0]) == 'weighted_maxcut':
        title = 'Weighted MaxCut'
    elif str(odata["hamiltonian"].iloc[0]) == 'rbm':
        title = 'Quantum RBMs'
    elif str(odata["hamiltonian"].iloc[0]) == 'heisenberg':
        title = 'Quantum spin glasses'
    elif str(odata["hamiltonian"].iloc[0]) == 'maxcut':
        title = 'MaxCut'

    plt.figure(figsize=(4,2.5))
    plt.title(title)
    plt.plot(nqubits_range, cv, 'o')
    plt.xlabel('$N$')
    if show_ylabel:
        plt.ylabel('Critical inverse temperature')

    def exponential(x, a, b):
        return a*np.exp(b*x)
    def base2(x, a, b):
        return a * 2**(b*x)
    def leandro(x, a, b, c):
        return c + a * np.exp(b*x)

    # pars, cov = curve_fit(f=exponential, xdata=nqubits_range, ydata=cv)
    # rmse = np.sqrt(np.mean(np.square(exponential(nqubits_range, *pars) - cv)))
    # plt.plot(nqubits_range, exponential(nqubits_range, *pars), ':', label=f'$\\beta_c(n) = {pars[0]:.2f} \cdot e^{{ {pars[1]:.2f} \cdot n}}$, RMSE={rmse:.2f}')

    # pars, cov = curve_fit(f=base2, xdata=nqubits_range, ydata=cv)
    # rmse2 = np.sqrt(np.mean(np.square(base2(nqubits_range, *pars) - cv)))
    # plt.plot(nqubits_range, base2(nqubits_range, *pars), '--', label=f'$\\beta_c(n) = {pars[0]:.2f} \cdot 2^{{ {pars[1]:.2f} \cdot n}}$, RMSE={rmse2:.2f}')

    pars, cov = curve_fit(f=leandro, xdata=nqubits_range, ydata=cv)
    rmse3 = np.sqrt(np.mean(np.square(leandro(nqubits_range, *pars) - cv)))
    if pars[2] >= 0:
        plt.plot(nqubits_range, leandro(nqubits_range, *pars), '--', label=f'${pars[0]:.2f} \cdot 2^{{ {pars[1]:.2f} \cdot N}} + {pars[2]:.2f}$\n(RMSE={rmse3:.2f})')
    else:
        plt.plot(nqubits_range, leandro(nqubits_range, *pars), '--', label=f'${pars[0]:.2f} \cdot 2^{{ {pars[1]:.2f} \cdot N}} {pars[2]:.2f}$\n(RMSE={rmse3:.2f})')

    plt.legend(frameon=False)

    output = str(odata["hamiltonian"].iloc[0]) + '_betac.pdf'
    print(f'Saving {output}')
    plt.savefig(f"{output}", bbox_inches='tight')


def do_fit(filenames):

    colors = ['C0', 'C1', 'C2']

    for index, filename in enumerate(filenames):
        data = pd.read_csv(filename)
        data = data.sort_values(by=['beta'])
        title = str(data["hamiltonian"].iloc[0]).replace('_', ' ') + ' hamiltonian'
        beta_range = data['beta'].unique()
        gdata = data.groupby(['beta'])
        means = gdata.mean()
        stds = gdata.std()
        if str(data["hamiltonian"].iloc[0]) == 'weighted_maxcut':
            title = 'Weighted MaxCut'
        elif str(data["hamiltonian"].iloc[0]) == 'rbm':
            title = 'Quantum RBMs'
        elif str(data["hamiltonian"].iloc[0]) == 'heisenberg':
            title = 'Quantum spin glasses'
        elif str(data["hamiltonian"].iloc[0]) == 'maxcut':
            title = 'MaxCut'

        if index == 0:
            fig, axs = plt.subplots(3, 1, figsize=(5,5), sharex=True)
            fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
            axs[0].set_title(title)

        def fun(x, a, b):
            return a * x ** b

        pars, cov = curve_fit(f=fun, xdata=beta_range, ydata=means['F_r'])
        rmse = np.sqrt(np.mean(np.square(fun(beta_range, *pars) - means['F_r'])))
        plot_band(axs[0], beta_range, means['F_r'], stds['F_r'], label=f'{str(data["nqubits"].iloc[0])} qubits')
        axs[0].plot(beta_range, fun(beta_range, *pars), '--', color=colors[index], label=f'${pars[0]:.2f} \cdot \\beta^{{ {pars[1]:.2f}}}$\n(RMSE={rmse:.2f})')

        axs[0].legend(frameon=False, ncol=3, prop={'size': 7})
        axs[0].set_ylabel('$r$ (uniform)')

        plot_band(axs[1], beta_range, means['F_fit_r'], stds['F_fit_r'])
        axs[1].set_ylabel('$r$ (optimized)')

        cv_params = np.zeros(shape=len(beta_range))
        std_params = np.zeros(shape=len(beta_range))
        for w, b in enumerate(beta_range):
            d = np.array([ float(i[1:-1]) for i in gdata.get_group(b)["F_fit_params"].tolist() ])
            cv_params[w] = np.mean(d)
            std_params[w] = np.std(d)

        pars, cov = curve_fit(f=fun, xdata=beta_range[1:], ydata=cv_params[1:])
        rmse = np.sqrt(np.mean(np.square(fun(beta_range[1:], *pars) - cv_params[1:])))

        plot_band(axs[2], beta_range[1:], cv_params[1:], std_params[1:])
        axs[2].plot(beta_range[1:], fun(beta_range[1:], *pars), ':', color=colors[index], label=f'${pars[0]:.2f} \cdot \\beta^{{ {pars[1]:.2f}}}$\n(RMSE={rmse:.2f})')

        axs[2].set_ylabel('$a$')
        axs[2].set_xlabel(r'$\beta$')
        axs[2].legend(loc=2, frameon=False, ncol=3, prop={'size': 7})

    output = filename.replace('.csv', '_fit.pdf').split('/')[-1]
    print(f'Saving {output}')
    plt.savefig(f"{output}", bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    do_beta('results/2002/weighted_maxcut/weighted_maxcut_qubits_12_maxbeta_2002_step_25_trials_1000.csv', show_ylabel=True)
    weighted_maxcut_files = [
        'results/2002/weighted_maxcut/weighted_maxcut_qubits_2_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/weighted_maxcut/weighted_maxcut_qubits_3_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/weighted_maxcut/weighted_maxcut_qubits_4_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/weighted_maxcut/weighted_maxcut_qubits_5_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/weighted_maxcut/weighted_maxcut_qubits_6_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/weighted_maxcut/weighted_maxcut_qubits_7_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/weighted_maxcut/weighted_maxcut_qubits_8_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/weighted_maxcut/weighted_maxcut_qubits_9_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/weighted_maxcut/weighted_maxcut_qubits_10_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/weighted_maxcut/weighted_maxcut_qubits_11_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/weighted_maxcut/weighted_maxcut_qubits_12_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/weighted_maxcut/weighted_maxcut_qubits_13_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/weighted_maxcut/weighted_maxcut_qubits_14_maxbeta_2002_step_25_trials_100.csv',
        'results/2002/weighted_maxcut/weighted_maxcut_qubits_15_maxbeta_2002_step_25_trials_100.csv'
        ]
    do_critical(weighted_maxcut_files, show_ylabel=False)
    do_fit([
        'results/2002/weighted_maxcut/weighted_maxcut_qubits_5_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/weighted_maxcut/weighted_maxcut_qubits_10_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/weighted_maxcut/weighted_maxcut_qubits_15_maxbeta_2002_step_25_trials_100.csv'
    ])
    # do_beta('results/2002/maxcut/maxcut_qubits_13_maxbeta_2002_step_25_trials_1000.csv')
    maxcut_files = [
        'results/2002/maxcut/maxcut_qubits_2_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/maxcut/maxcut_qubits_3_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/maxcut/maxcut_qubits_4_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/maxcut/maxcut_qubits_5_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/maxcut/maxcut_qubits_6_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/maxcut/maxcut_qubits_7_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/maxcut/maxcut_qubits_8_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/maxcut/maxcut_qubits_9_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/maxcut/maxcut_qubits_10_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/maxcut/maxcut_qubits_11_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/maxcut/maxcut_qubits_12_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/maxcut/maxcut_qubits_13_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/maxcut/maxcut_qubits_14_maxbeta_2002_step_25_trials_100.csv',
        'results/2002/maxcut/maxcut_qubits_15_maxbeta_2002_step_25_trials_100.csv'
        ]
    do_critical(maxcut_files, show_ylabel=True)
    do_beta('results/2002/rbm/rbm_qubits_12_maxbeta_2002_step_25_trials_1000.csv', show_ylabel=False)
    rbm_files = [
        'results/2002/rbm/rbm_qubits_2_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/rbm/rbm_qubits_3_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/rbm/rbm_qubits_4_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/rbm/rbm_qubits_5_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/rbm/rbm_qubits_6_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/rbm/rbm_qubits_7_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/rbm/rbm_qubits_8_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/rbm/rbm_qubits_9_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/rbm/rbm_qubits_10_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/rbm/rbm_qubits_11_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/rbm/rbm_qubits_12_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/rbm/rbm_qubits_13_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/rbm/rbm_qubits_14_maxbeta_2002_step_25_trials_100.csv'
        ]
    do_critical(rbm_files, show_ylabel=True)
    do_beta('results/2002/heisenberg/heisenberg_qubits_12_maxbeta_2002_step_25_trials_1000.csv', show_ylabel=False)
    heisenberg_files = [
        'results/2002/heisenberg/heisenberg_qubits_2_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/heisenberg/heisenberg_qubits_3_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/heisenberg/heisenberg_qubits_4_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/heisenberg/heisenberg_qubits_5_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/heisenberg/heisenberg_qubits_6_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/heisenberg/heisenberg_qubits_7_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/heisenberg/heisenberg_qubits_8_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/heisenberg/heisenberg_qubits_9_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/heisenberg/heisenberg_qubits_10_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/heisenberg/heisenberg_qubits_11_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/heisenberg/heisenberg_qubits_12_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/heisenberg/heisenberg_qubits_13_maxbeta_2002_step_25_trials_1000.csv',
        'results/2002/heisenberg/heisenberg_qubits_14_maxbeta_2002_step_25_trials_100.csv',
        'results/2002/heisenberg/heisenberg_qubits_15_maxbeta_2002_step_25_trials_100.csv'
        ]
    do_critical(heisenberg_files, skip=True, show_ylabel=False)
    # do_beta('results/2002/heisenberg_fully_connected/heisenberg_fully_connected_qubits_12_maxbeta_2002_step_25_trials_1000.csv')
    # heisenberg_fully_connected_files = [
    #     'results/2002/heisenberg_fully_connected/heisenberg_fully_connected_qubits_2_maxbeta_2002_step_25_trials_1000.csv',
    #     'results/2002/heisenberg_fully_connected/heisenberg_fully_connected_qubits_3_maxbeta_2002_step_25_trials_1000.csv',
    #     'results/2002/heisenberg_fully_connected/heisenberg_fully_connected_qubits_4_maxbeta_2002_step_25_trials_1000.csv',
    #     'results/2002/heisenberg_fully_connected/heisenberg_fully_connected_qubits_5_maxbeta_2002_step_25_trials_1000.csv',
    #     'results/2002/heisenberg_fully_connected/heisenberg_fully_connected_qubits_6_maxbeta_2002_step_25_trials_1000.csv',
    #     'results/2002/heisenberg_fully_connected/heisenberg_fully_connected_qubits_7_maxbeta_2002_step_25_trials_1000.csv',
    #     'results/2002/heisenberg_fully_connected/heisenberg_fully_connected_qubits_8_maxbeta_2002_step_25_trials_1000.csv',
    #     'results/2002/heisenberg_fully_connected/heisenberg_fully_connected_qubits_9_maxbeta_2002_step_25_trials_1000.csv',
    #     'results/2002/heisenberg_fully_connected/heisenberg_fully_connected_qubits_10_maxbeta_2002_step_25_trials_1000.csv',
    #     'results/2002/heisenberg_fully_connected/heisenberg_fully_connected_qubits_11_maxbeta_2002_step_25_trials_1000.csv',
    #     'results/2002/heisenberg_fully_connected/heisenberg_fully_connected_qubits_12_maxbeta_2002_step_25_trials_1000.csv',
    #     'results/2002/heisenberg_fully_connected/heisenberg_fully_connected_qubits_13_maxbeta_2002_step_25_trials_1000.csv',
    #     'results/2002/heisenberg_fully_connected/heisenberg_fully_connected_qubits_14_maxbeta_2002_step_25_trials_100.csv',
    #     'results/2002/heisenberg_fully_connected/heisenberg_fully_connected_qubits_15_maxbeta_2002_step_25_trials_100.csv'
    #     ]
    # do_critical(heisenberg_fully_connected_files, skip=True, show_ylabel=False)
