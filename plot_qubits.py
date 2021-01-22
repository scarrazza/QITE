import argparse
import pandas as pd
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{siunitx}')
plt.rc('font', family='serif', size=10)
plt.rc('pdf', compression=9)
plt.rc('axes', labelsize='large')


parser = argparse.ArgumentParser()
parser.add_argument("filename", nargs='+', type=str)
parser.add_argument("--beta", default=202, type=int)
parser.add_argument("--title", default='', type=str)
args = vars(parser.parse_args())


def plot_band(axs, x, cv, std, label=None):
    axs.plot(x, cv, '.-',label=label)
    axs.fill_between(x, cv-std, cv+std, alpha=0.1)


def main(filename, title, beta):
    """Main function for simulation.
    """
    data = pd.DataFrame()
    for f in filename:
        data = data.append(pd.read_csv(f))

    if beta not in data['beta'].unique():
        raise RuntimeError("Requested beta not available.")

    data = data[data['beta'] == beta]
    nqubits_range = data['nqubits'].unique()
    gdata = data.groupby(['nqubits'])
    means = gdata.mean()
    stds = gdata.std()

    fig, axs = plt.subplots(3, 1, figsize=(5,7), sharex=True)
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    axs[0].set_title(title)
    plot_band(axs[0], nqubits_range, means['F'], stds['F'], label='F linear')
    plot_band(axs[0], nqubits_range, means['F_fit'], stds['F_fit'], label='F fit')
    plot_band(axs[0], nqubits_range, means['AA'], stds['AA'], label='AA')
    plot_band(axs[0], nqubits_range, means['C'], stds['C'], label='C')

    axs[0].set_yscale('log')
    axs[0].legend(frameon=False, ncol=2)
    axs[0].set_ylabel('Query complexity')

    plot_band(axs[1], nqubits_range, means['F']/means['F'], stds['F']/means['F'])
    plot_band(axs[1], nqubits_range, means['F_fit']/means['F'], stds['F_fit']/means['F'])

    plot_band(axs[1], nqubits_range, means['AA']/means['F'], stds['AA']/means['F'])
    plot_band(axs[1], nqubits_range, means['C']/means['F'], stds['C']/means['F'])

    axs[1].set_ylim([0,2])
    axs[1].set_ylabel('Ratio to F linear');

    axs[2].legend(frameon=False, ncol=2)
    axs[2].set_xlabel(r'nqubits')
    axs[2].set_ylabel('Best $r$')

    plot_band(axs[2], nqubits_range, means['F_r'], stds['F_r'], label='F linear')
    plot_band(axs[2], nqubits_range, means['F_fit_r'], stds['F_fit_r'], 'F fit')

    output = filename[0].replace('.csv','_qubits.pdf')
    print(f'Saving {output}')
    plt.savefig(f"{output}", bbox_inches='tight')


if __name__ == "__main__":
    main(**args)
