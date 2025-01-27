import math
import scipy
from scipy import sparse
from scipy.sparse import csr_matrix
import operators
import common_functions

def delta_h4(lmax, e_max, m, r, basis):
    term4 = operators.term_4low(lmax, e_max, m, r, basis)
    term3 = operators.term_1raise3low(lmax, e_max, m, r, basis)
    h4mat = term4 + term4.transpose() + operators.term_2raise2low(lmax, e_max, m, r, basis) + term3 + term3.transpose()
    return h4mat
def delta_h2(lmax, e_max, m, r, basis):
    term2 = operators.term_2low(lmax, e_max, m, r, basis)
    return term2 + term2.transpose() + operators.term_1raise1low(lmax, e_max, m, r, basis)
def h0(lmax, e_max, m, r, basis):
    # start=time.time()
    length_basis = len(basis) 
    row = []
    col = []
    data = []
    for state_index in range(length_basis):
        state = basis[state_index]
        row.append(state_index)
        col.append(state_index)
        data.append(common_functions.state_energy(state, m, r))
    return csr_matrix((data, (row, col)), shape=(length_basis, length_basis))

def h_full(lmax, e_max, m, delta_m, basis, g, r):
    if pow(m, 2) < pow(e_max, 2) / 4:
        lmaxeff = int(min(lmax, math.floor(math.sqrt(pow(r, 2) * (float(pow(e_max,2))/4. - pow(m, 2))))))
    else:
        lmaxeff = 0
    return h0(lmaxeff, e_max, m, r, basis) + 1./4.*delta_m*delta_h2(lmaxeff, e_max, m, r, basis) +g/r*1/(8*math.pi) *1/24.*delta_h4(lmax, e_max, m, r, basis)
#
#

