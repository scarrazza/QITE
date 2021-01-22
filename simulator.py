import argparse
import hamiltonians
import numpy as np
import pandas as pd
from config import K
from fquite import FragmentedQuITE


parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default=2, type=int)
parser.add_argument("--hamiltonian", default="maxcut", type=str)
parser.add_argument("--maxbeta", default=300, type=int)
parser.add_argument("--maxr", default=25, type=int)
parser.add_argument("--trial", default=0, type=int)
parser.add_argument("--output", default='./', type=str)
args = vars(parser.parse_args())


def main(nqubits, hamiltonian, maxbeta, maxr, trial, output):
    """Main function for simulation."""

    # load hamiltonian
    np.random.seed(trial)
    h = getattr(hamiltonians, hamiltonian)(nqubits)
    energy = K.linalg.eigvalsh(h)
    energy /= ((K.max(energy) - K.min(energy))/2)
    del h

    # build frag quite obj
    frag = FragmentedQuITE(nqubits, energy)

    # TODO move to runcard
    beta_range = range(2, maxbeta+1, 10)
    r_range = range(2, maxr)

    result = []
    for beta in beta_range:
        psuc = frag.Psuc(beta)
        aa = frag.AA(beta, psuc)
        c = frag.C(beta, psuc)
        f, f_r_best, f_depth = frag.rF(beta, r_range)
        f_fit, f_fit_r_best, f_fit_depth = frag.rFfit(beta, r_range)
        obj = {
            'beta': beta,
            'trial': trial,
            'AA': aa,
            'Psuc': psuc,
            'C': c,
            'C_depth': c * psuc,
            'F': f,
            'F_depth': f_depth,
            'F_r': f_r_best,
            'F_fit': f_fit,
            'F_fit_depth': f_fit_depth,
            'F_fit_r': f_fit_r_best
        }
        result.append(obj)

    df = pd.DataFrame(result)
    with open(f'{output}/{hamiltonian}_{nqubits}_{maxbeta}_{maxr}.csv', 'a') as f:
        df.to_csv(f, header=f.tell()==0)


if __name__ == "__main__":
    print(args)
    main(**args)
