# HTET_NLO
Hamiltonian truncation is a non-perturbative method to calculate observables in QFT, and this is achieved by truncating an infinite dimensional hamiltonian to a finite dimensional one, with the help of a cut off energy E_max which gives rise to an effective hamiltonian. The effective Hamiltonian can be computed by matching a transition amplitude to the full theory, and gives corrections order by order as an expansion in powers of 1/Emax. The effective Hamiltonian is non-local, with the non-locality controlled in an expansion in powers of H0/Emax

 [Previous Paper on 2D-phi^4 Leading Order](https://arxiv.org/pdf/2110.08273).


# How to Use? 
The usage of the code is pretty basic. An example can be found in _example.py_. 

# _TODO_
We plan to use decoders as much as possible to speed up the code, as at its current state, for large volume-high energy cutoff, it can take a lot of time to generate hamiltonians and the basis. This is why we usually save the matrices we generate in the forms of npz and txt. If you do not wish to save them, please remove them from example.py. It must be noted that the simulation requires a large piece of memory for the mentioned cases. For most of the numerical anaylsis in the paper, we have used r ~ 3.1, and Emax ~ 18-19, with a 32 GB RAM laptop, for the high volume case. Low volume case uses Emax ~ 27, and r ~ 1.6 

# Contrubitors
Feel free to reach to :
- Ekrem Orhan Demiray - eo.demiray@ufl.edu / eodemiray@gmail.com
- Rachel Houtz - rachel.houtz@ufl.edu
- Kara Farnsworth - kmfarnsworth@gmail.com
