import argparse
import hamiltonians
import numpy as np
import pandas as pd
from fquite import FragmentedQuITE
from multiprocessing import Pool
from config import K, dtype


parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default=2, type=int)
parser.add_argument("--hamiltonian", default="maxcut", type=str)
parser.add_argument("--maxbeta", default=300, type=int)
parser.add_argument("--maxr", default=25, type=int)
parser.add_argument("--trials", default=1, type=int)
parser.add_argument("--output", default='./', type=str)
parser.add_argument("--processes", default=1, type=int)
parser.add_argument("--gpu", action='store_true')
args = vars(parser.parse_args())


def get_energy(nqubits, hamiltonian, trial):
    print(f'Hamiltonian: {hamiltonian} - trial: {trial} - nqubits: {nqubits}')
    np.random.seed(trial)
    h = np.asarray(getattr(hamiltonians, hamiltonian)(nqubits))
    print(h, h.dtype)
    energy = np.linalg.eigvalsh(h)
    energy /= ((np.max(energy) - np.min(energy))/2)
    del h
    return energy


def run(nqubits, hamiltonian, maxbeta, maxr, trial, energy=None):
    # load hamiltonian
    if energy is None:
        energy = get_energy(nqubits, hamiltonian, trial)

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
        f_fit, f_fit_r_best, f_fit_depth, params = frag.rFfit(beta, r_range)
        obj = {
            'hamiltonian': hamiltonian,
            'nqubits': nqubits,
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
            'F_fit_r': f_fit_r_best,
            'F_fit_params': params
        }
        result.append(obj)
    return result


def main(nqubits, hamiltonian, maxbeta, maxr, trials, output, processes, gpu):
    """Main function for simulation.
    """
    if not gpu:
        jobs = [(nqubits, hamiltonian, maxbeta, maxr, t) for t in range(trials)]
        with Pool(processes=processes) as p:
            results = p.starmap(run, jobs)
    else:
        pool = Pool(processes=processes)
        results = []
        for t in range(trials):
            energy = get_energy(nqubits, hamiltonian, t)
            pool.apply_async(
                run, [nqubits, hamiltonian, maxbeta, maxr, t, energy],
                callback=results.append)
        pool.close()
        pool.join()
    df = pd.DataFrame()
    for result in results:
        df = df.append(result)
    filename = f'{output}/{hamiltonian}_qubits_{nqubits}_maxbeta_{maxbeta}_maxr_{maxr}_trials_{trials}.csv'
    with open(filename, 'a') as f:
        df.to_csv(f, header=f.tell()==0)


if __name__ == "__main__":
    main(**args)
