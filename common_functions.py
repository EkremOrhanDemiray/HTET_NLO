import math
import numpy as np
import scipy
from scipy import sparse
from scipy.stats import uniform
from scipy.sparse import csr_matrix
from scipy.sparse import identity
import scipy.sparse.linalg as sparse_la
import time
import os
import os.path
import bisect
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from numba import jit
from numba.extending import overload

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from IPython.display import display, clear_output
from numba import njit
preset_colors=mcolors.cnames
color_names_all = sorted(preset_colors.keys())
i=0
bad_colors=[0, 1, 2, 3, 4, 5, 6, 8, 12, 14, 18, 46, 20, 24, 26, 27, 49, 50, 55, 57, 61, 62, 63, 64, 65, 66,67, 68, 69, 70, 71,72,73, 80, 81, 82, 84,93, 97, 98, 99, 100, 102, 108,109, 110,112,113,115,117,127,129, 134,135,139,143, 144, 145, 146]

    
color_names= color_names_all[32:33]+ color_names_all[120:121]
color_names= color_names +color_names_all[106:107]+color_names_all[105:106]+color_names_all[51:52]
color_names= color_names + color_names_all[147:148]+color_names_all[54:55]+color_names_all[138:139]+color_names_all[21:22]
color_names= color_names +color_names_all[60:61]+color_names_all[118:119]+color_names_all[95:96]+color_names_all[40:41]
color_names= color_names +color_names_all[53:54]+color_names_all[7:8]


PATH = 'datafiles/'
dir_name = "{}".format(PATH)
if not os.path.isdir(dir_name): os.mkdir(dir_name)
basis_dir_name = "{}".format(PATH) + "BasisData/"
basis_dir_name_write = "{}".format(PATH) + "BasisData_HTET/"
if not os.path.isdir(basis_dir_name_write): os.mkdir(basis_dir_name_write)
omega_list = []

omegaList = []
def gen_omega_list_Ekrem(lmax, m, r):
    global omegaList
    omegaList = [omega(i,m,r) for i in range(lmax+1)]
    
def omega(l, m, r): 
    output = math.sqrt(float(l) * float(l) / (r*r) + m*m)
    return output

def gen_omega_list(lmax, m, r):
    global omega_list
    omega_list = []
    for l in range(lmax+1):
        omega_list.append(omega(l, m, r))

def gen_basis(lmax, e_max, m, r):
    result = []
    if lmax == 0:
        result = list(map(lambda x: [x], list(range(0, int(math.floor(e_max/m)) + 1))))
    else:
        for n in range(0, int(math.floor(e_max/omega(lmax, m, r))) + 1):
            prev_list = gen_basis(lmax-1, e_max-n * omega(lmax, m, r), m, r)
            for NP in range(0, n+1):
                result = result + list(map(lambda x: list(x) + [NP, n-NP], prev_list))
    return result

def gen_basis_aLittleFaster(lmax, e_max, m, r):
    result = []
    if lmax == 0:
        result = list(map(lambda x: [x], list(range(0, int(math.floor(e_max/m)) + 1))))
    else:
        for n in range(0, int(math.floor(e_max/omega(lmax, m, r))) + 1):
            prev_list = gen_basis(lmax-1, e_max-n * omega(lmax, m, r), m, r)

            result = [(result + list(map(lambda x: list(x) + [NP, n-NP], prev_list))) for NP in range(0, n+1)] 
    return result

def make_basis(lmax, e_max, m, r):
    global omega_list
    basis = gen_basis(lmax, e_max, m, r)
    gen_omega_list(lmax, m, r)
    return basis


def ix(l, sigma):
    if l == 0:
        return 0
    elif sigma >= 0:
        return 2*l-1
    else: 
        return 2*l


def the_l(ix_arg):
    if ix_arg == 0:
        return 0
    elif ix_arg % 2 != 0:
        return int((ix_arg+1) / 2)
    else:
        return int(ix_arg/2)


def the_sigma(ix_arg):
    if ix_arg == 0:
        return 0
    elif ix_arg % 2 != 0:
        return 1
    else:
        return -1


def count_particles(state):
    num_particles = 0
    for particles in state:
        num_particles = num_particles+particles
    return num_particles


def l_total(state):
    l_sum = 0
    for i in range(1, len(state)):
        if i % 2 != 0:
            l_sum = l_sum + state[i] * the_l(i)
        else:
            l_sum = l_sum-state[i] * the_l(i)
    return l_sum


def state_energy(state, m, r):
    global omega_list
    total_energy = 0
    stateLen = len(state)
    for i in range(stateLen):
        total_energy += state[i] * omega_list[abs(the_l(i))]
    return total_energy


def find_pos(state, basis, min_index, max_index):
    index = bisect.bisect_left(basis, state, min_index, max_index)
    if basis[index] == state:
        return index
    else:
        return -1



def basis_odd(basis):
    new_basis = []
    for state in basis:
        if sum(state) % 2 != 0:
            new_basis.append(state)
    return new_basis


def basis_even(basis):
    new_basis = []
    for state in basis:
        if sum(state) % 2 == 0:
            new_basis.append(state)
    return new_basis



def basis_l0(basis):
    new_basis = []
    for state in basis:
        if l_total(state) == 0:
            new_basis.append(state)
    return new_basis



def basis_nmax(basis, n_max):
    new_basis = []
    for state in basis:
        if count_particles(state) <= n_max:
            new_basis.append(state)
    return new_basis