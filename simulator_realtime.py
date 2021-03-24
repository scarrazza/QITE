import argparse
import hamiltonians
import numpy as np
import pandas as pd
from fquite_realtime import FragmentedQuITE, gamma_opt
from multiprocessing import Pool
from config import K, dtype


parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default=2, type=int)
parser.add_argument("--hamiltonian", default="maxcut", type=str)
parser.add_argument("--maxbeta", default=300, type=int)
parser.add_argument("--step", default=10, type=int)
parser.add_argument("--trials", default=1, type=int)
parser.add_argument("--output", default='./', type=str)
parser.add_argument("--processes", default=1, type=int)
parser.add_argument("--gpu", action='store_true', help='computes hamiltonian in async with frag.')
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


def run(nqubits, hamiltonian, maxbeta, step, trial, energy=None):
    # load hamiltonian
    if energy is None:
        energy = get_energy(nqubits, hamiltonian, trial)

    # build frag quite obj
    frag = FragmentedQuITE(nqubits, energy)

    # TODO move to runcard
    beta_range = range(2, maxbeta+1, step)

    result = []
    for beta in beta_range:
        gammaAA = gamma_opt(beta, AA=True)
        gamma = gamma_opt(beta)
        psucAA = frag.Psuc(beta, gammaAA)
        psuc = frag.Psuc(beta, gamma)
        aa = frag.AA(beta, psucAA, gammaAA)
        c = frag.C(beta, psuc, gamma)
        f, f_r_best, f_depth = frag.rF(beta, gamma)
        f_fit, f_fit_r_best, f_fit_depth, params = frag.rFfit(beta, gamma)
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


def main(nqubits, hamiltonian, maxbeta, step, trials, output, processes, gpu):
    """Main function for simulation.
    """
    if not gpu:
        jobs = [(nqubits, hamiltonian, maxbeta, step, t) for t in range(trials)]
        with Pool(processes=processes) as p:
            results = p.starmap(run, jobs)
    else:
        pool = Pool(processes=processes)
        results = []
        for t in range(trials):
            energy = get_energy(nqubits, hamiltonian, t)
            pool.apply_async(
                run, [nqubits, hamiltonian, maxbeta, step, t, energy],
                callback=results.append)
        pool.close()
        pool.join()
    df = pd.DataFrame()
    for result in results:
        df = df.append(result)
    filename = f'{output}/{hamiltonian}_qubits_{nqubits}_maxbeta_{maxbeta}_step_{step}_trials_{trials}.csv'
    with open(filename, 'a') as f:
        df.to_csv(f, header=f.tell()==0)


if __name__ == "__main__":
    main(**args)
