#!/bin/bash

for i in {2..12}; do python simulator.py --nqubits $i --hamiltonian maxcut --maxbeta 300 --maxr 45 --trials 1000 --processes 32; done
for i in {2..12}; do python simulator.py --nqubits $i --hamiltonian weighted_maxcut --maxbeta 500 --maxr 45 --trials 1000 --processes 32; done
for i in {2..12}; do python simulator.py --nqubits $i --hamiltonian rbm --maxbeta 300 --maxr 45 --trials 1000 --processes 32; done
