import math
import numpy as np
import scipy
from scipy import sparse
from scipy.stats import uniform
from scipy.sparse import csr_matrix
from scipy.sparse import identity
import scipy.sparse.linalg as sparse_la
import os.path
import bisect
import common_functions


def lower4(l1, s1, l2, s2, l3, s3, l4, s4, e_max, m, r, basis):
    row = []
    col = []
    data = []
    norm = 0
    ix_max = len(basis[0])
    length_basis = len(basis)
    for state_index in range(length_basis):
        pos = -1
        state=basis[state_index]
        norm = math.sqrt(state[common_functions.ix(l1, s1)])
        if norm != 0:
            newstate = [state[common_functions.ix(l1, s1)] - 1 if i == common_functions.ix(l1, s1) else state[i] for i in range(ix_max)]
            norm = norm*math.sqrt(newstate[common_functions.ix(l2, s2)])
            if norm != 0:
                newstate2 = [newstate[common_functions.ix(l2, s2)] - 1 if i == common_functions.ix(l2, s2) else newstate[i] for i in range(ix_max)]
                norm = norm*math.sqrt(newstate2[common_functions.ix(l3, s3)])
                if norm != 0:
                    newstate3 = [newstate2[common_functions.ix(l3, s3)] - 1 if i == common_functions.ix(l3, s3) else newstate2[i] for i in range(ix_max)]
                    norm = norm*math.sqrt(newstate3[common_functions.ix(l4, s4)])
                    if norm != 0:
                        newstate4 = [newstate3[common_functions.ix(l4, s4)] - 1 if i == common_functions.ix(l4, s4) else newstate3[i] for i in range(ix_max)]
                        if common_functions.state_energy(newstate4, m, r) <= e_max:
                            pos = bisect.bisect_left(basis, newstate4)
        if pos != -1:
            row.append(state_index)
            col.append(pos)
            data.append(norm)
    return csr_matrix((data, (row, col)), shape=(length_basis, length_basis))

def raise1low3(l1, s1, l2, s2, l3, s3, l4, s4, e_max, m, r, basis):
    # l4, s4 is the raising operator
    row = []
    col = []
    data = []
    norm = 0
    ix_max = len(basis[0])
    length_basis = len(basis)
    for state_index in range(length_basis):
        pos = -1
        state = basis[state_index]
        norm = math.sqrt(state[common_functions.ix(l1, s1)])
        if norm != 0:
            newstate = [state[common_functions.ix(l1, s1)] - 1 if i == common_functions.ix(l1, s1) else state[i] for i in range(ix_max)]
            norm = norm*math.sqrt(newstate[common_functions.ix(l2, s2)])
            if norm != 0:
                newstate2 = [newstate[common_functions.ix(l2, s2)] - 1 if i == common_functions.ix(l2, s2) else newstate[i] for i in range(ix_max)]
                norm = norm*math.sqrt(newstate2[common_functions.ix(l3, s3)])
                if norm != 0:
                    newstate3 = [newstate2[common_functions.ix(l3, s3)] - 1 if i == common_functions.ix(l3, s3) else newstate2[i] for i in range(ix_max)]
                    norm = norm*math.sqrt(newstate3[common_functions.ix(l4, s4)] + 1)
                    if norm != 0:
                        newstate4 = [newstate3[common_functions.ix(l4, s4)] + 1 if i == common_functions.ix(l4, s4) else newstate3[i] for i in range(ix_max)]
                        if common_functions.state_energy(newstate4, m, r) <= e_max:
                            pos = bisect.bisect_left(basis, newstate4)
        if pos != -1:
            row.append(state_index)
            col.append(pos)
            data.append(norm)
    return csr_matrix((data, (row, col)), shape=(length_basis, length_basis))

def raise2low2(l1, s1, l2, s2, l3, s3, l4, s4, e_max, m, r, basis):
    row = []
    col = []
    data = []
    norm = 0
    ix_max = len(basis[0])
    length_basis = len(basis)
    pos = -1
    for state_index in range(length_basis):
        state=basis[state_index]
        norm = math.sqrt(state[common_functions.ix(l1, s1)])
        if norm != 0:
            newstate = [state[common_functions.ix(l1, s1)] - 1 if i == common_functions.ix(l1, s1) else state[i] for i in range(ix_max)]
            norm = norm*math.sqrt(newstate[common_functions.ix(l2, s2)])
            if norm != 0:
                newstate2 = [newstate[common_functions.ix(l2, s2)] - 1 if i == common_functions.ix(l2, s2) else newstate[i] for i in range(ix_max)]
                norm = norm*math.sqrt(newstate2[common_functions.ix(l3, s3)]+1)
                if norm != 0:
                    newstate3 = [newstate2[common_functions.ix(l3, s3)] + 1 if i == common_functions.ix(l3, s3) else newstate2[i] for i in range(ix_max)]
                    norm = norm*math.sqrt(newstate3[common_functions.ix(l4, s4)] + 1)
                    if norm != 0:
                        newstate4 = [newstate3[common_functions.ix(l4, s4)] + 1 if i == common_functions.ix(l4, s4) else newstate3[i] for i in range(ix_max)]
                        if newstate4==state:
                            pos = state_index
                        elif common_functions.state_energy(newstate4, m, r) <= e_max:
                            pos = bisect.bisect_left(basis, newstate4)
        if pos != -1:
            row.append(state_index)
            col.append(pos)
            data.append(norm)
            pos=-1
    return csr_matrix((data, (row, col)), shape=(length_basis, length_basis))

def lower2(l1, s1, l2, s2, e_max, m, r, basis):
    row = []
    col = []
    data = []
    norm = 0
    ix_max = len(basis[0])
    length_basis = len(basis)
    for state_index in range(length_basis):
        pos = -1
        state = basis[state_index]
        norm = math.sqrt(state[common_functions.ix(l1, s1)])
        if norm != 0:
            newstate = [state[common_functions.ix(l1, s1)] - 1 if i == common_functions.ix(l1, s1) else state[i] for i in range(ix_max)]
            norm = norm*math.sqrt(newstate[common_functions.ix(l2, s2)])
            if norm != 0:
                newstate2 = [newstate[common_functions.ix(l2, s2)] - 1 if i == common_functions.ix(l2, s2) else newstate[i] for i in range(ix_max)]
                if common_functions.state_energy(newstate2, m, r) <= e_max:
                    pos = bisect.bisect_left(basis, newstate2)
        if pos != -1:
            row.append(state_index)
            col.append(pos)
            data.append(norm)
    return csr_matrix((data, (row, col)), shape=(length_basis, length_basis))
def raise1low1(l1, s1, l2, s2, e_max, m, r, basis):
    row = []
    col = []
    data = []
    norm = 0
    ix_max = len(basis[0])
    length_basis=len(basis)
    for state_index in range(length_basis):
        pos = -1
        state=basis[state_index]
        norm = math.sqrt(state[common_functions.ix(l1, s1)])
        if norm != 0:
            newstate = [state[common_functions.ix(l1, s1)] - 1 if i == common_functions.ix(l1, s1) else state[i] for i in range(ix_max)]
            norm = norm*math.sqrt(newstate[common_functions.ix(l2, s2)] + 1)
            if norm != 0:
                newstate2 = [newstate[common_functions.ix(l2, s2)] + 1 if i == common_functions.ix(l2, s2) else newstate[i] for i in range(ix_max)]
                if newstate2==state:
                    pos=state_index
                elif common_functions.state_energy(newstate2, m, r) <= e_max:
                    pos = bisect.bisect_left(basis, newstate2)
        if pos != -1:
            row.append(state_index)
            col.append(pos)
            data.append(norm)
    return csr_matrix((data, (row, col)), shape=(length_basis, length_basis))

def raise4_op_list(lmax, e_max, m, r):
    op_list = []
# -lmax <= j < k < l < n <= lmax
    for j in range(-lmax, lmax-2):
        for k in range(j+1, lmax-1):
            for l in range(k+1, lmax):
                n = -l-k-j
                if n in range(l+1, lmax+1) and common_functions.omega_list[abs(j)] + common_functions.omega_list[abs(k)] +common_functions.omega_list[abs(l)] + common_functions.omega_list[abs(n)] <= e_max:
                    op = [j, k, l, n]
                    op_list.append([op, 24])
# -lmax <= j < k < l = n <= lmax
    for j in range(-lmax, lmax-1):
        for k in range(j+1, lmax):
            if (-k-j) % 2 == 0:
                l = int((-k-j)/2)
                if l in range(k+1, lmax+1) and common_functions.omega_list[abs(j)] + common_functions.omega_list[abs(k)] + 2 * common_functions.omega_list[abs(l)] <= e_max:
                    op = [j, k, l, l]
                    op_list.append([op, 12])
# -lmax <= j < k = n < l <= lmax
    for j in range(-lmax, lmax-1):
        for k in range(j+1, lmax):
            l = -2 * k-j
            if l in range(k+1, lmax+1) and common_functions.omega_list[abs(j)] + 2 * common_functions.omega_list[abs(k)] + common_functions.omega_list[abs(l)] <= e_max:
                op = [j, k, k, l]
                op_list.append([op, 12])
# -lmax <= j = n < k < l <= lmax
    for j in range(-lmax, lmax-1):
        for k in range(j+1, lmax):
            l = -k-2 * j
            if l in range(k+1, lmax+1) and 2 * common_functions.omega_list[abs(j)] + common_functions.omega_list[abs(k)] + common_functions.omega_list[abs(l)] <= e_max:
                op = [j, j, k, l]
                op_list.append([op, 12])
# -lmax <= j = k < l= n <= lmax
    for k in range(-lmax, lmax):
        l = -k
        if l in range(k+1, lmax+1) and 2 * common_functions.omega_list[abs(k)] + 2 * common_functions.omega_list[abs(l)] <= e_max:
            op = [k, k, l, l]
            op_list.append([op, 6])
# -lmax <= j < k = l = n <= lmax
    for j in range(-lmax, lmax):
        if j % 3 == 0:
            l = - j / 3
            l = int(l)
            if l in range(j+1, lmax+1) and common_functions.omega_list[abs(j)] + 3 * common_functions.omega_list[abs(l)] <= e_max:
                op = [j, l, l, l]
                op_list.append([op, 4])
# -lmax <= j  = l = n < k <= lmax
    for j in range(-lmax, lmax):
        k = -3 * j
        if k in range(j+1, lmax+1) and 3 * common_functions.omega_list[abs(j)] + common_functions.omega_list[abs(k)] <= e_max:
            op = [j, j, j, k]
            op_list.append([op, 4])
# -lmax <= j = k = l = n <= lmax
    op = [0, 0, 0, 0]
    op_list.append([op, 1])
    return op_list


def raise31_op_list(lmax, e_max, m, r):
    op_list = []
    # -lmax <= j < k < l <= lmax
    for j in range(-lmax, lmax-1):
        for k in range(j+1, lmax):
            for l in range(k+1, lmax+1):
                n = l + k + j
                if n in range(-lmax, lmax+1) and common_functions.omega_list[abs(j)] + common_functions.omega_list[abs(k)] +common_functions.omega_list[abs(l)] <= e_max and common_functions.omega_list[abs(n)] <= e_max:
                    op = [j, k, l, n]
                    op_list.append([op, 6])
    # -lmax <= j < k = l <= lmax
    for j in range(-lmax, lmax):
        for k in range(j+1, lmax+1):
            n = j + 2 * k
            if n in range(-lmax, lmax+1) and common_functions.omega_list[abs(j)] + 2 * common_functions.omega_list[abs(k)] <= e_max and common_functions.omega_list[abs(n)] <= e_max:
                op = [j, k, k, n]
                op_list.append([op, 3])
    # -lmax <= j = k < l <= lmax
    for k in range(-lmax, lmax):
        for l in range(k+1, lmax+1):
            n = 2 * k + l
            if n in range(-lmax, lmax+1) and 2 * common_functions.omega_list[abs(k)] + common_functions.omega_list[abs(l)] <= e_max and common_functions.omega_list[abs(n)] <= e_max:
                op = [k, k, l, n]
                op_list.append([op, 3])
    # -lmax <= j = k = l <= lmax
    for k in range(math.ceil(-lmax/3), math.floor(lmax/3) + 1):
        n = 3 * k
        if n in range(-lmax, lmax+1) and 3 * common_functions.omega_list[abs(k)] <= e_max and common_functions.omega_list[abs(n)] <= e_max:
            op = [k, k, k, n]
            op_list.append([op, 1])
    return op_list

def raise22_op_list(lmax, e_max, m, r):
    op_list = []
    # -lmax <= j < k <= lmax
    # -lmax <= l < n <= lmax
    for j in range(-lmax, lmax):
        for k in range(j+1, lmax+1):
            for l in range(-lmax, lmax):
                n = -l + k + j
                if n in range(l+1, lmax+1) and common_functions.omega_list[abs(j)] + common_functions.omega_list[abs(k)] <= e_max and common_functions.omega_list[abs(l)] + common_functions.omega_list[abs(n)] <= e_max:
                    op = [j, k, l, n]
                    op_list.append([op, 4])
    # -lmax <= j < k <= lmax
    # -lmax <= l = n <= lmax
    for j in range(-lmax, lmax):
        for k in range(j+1, lmax+1):
            if (k+j) % 2 == 0:
                l = int((k+j)/2)
                if l in range(-lmax, lmax+1) and common_functions.omega_list[abs(j)] + common_functions.omega_list[abs(k)] <= e_max and 2 * common_functions.omega_list[abs(l)] <= e_max:
                    op = [j, k, l, l]
                    op_list.append([op, 2])
    # -lmax <= j = k <= lmax
    # -lmax <= l < n <= lmax
    for l in range(-lmax, lmax):
        for n in range(l+1, lmax+1):
            if (l+n) % 2 == 0:
                k = int((l+n)/2)
                if k in range(-lmax, lmax+1) and 2 * common_functions.omega_list[abs(k)] <= e_max and common_functions.omega_list[abs(l)] + common_functions.omega_list[abs(n)] <= e_max:
                    op = [k, k, l, n]
                    op_list.append([op, 2])
    # -lmax <= j = k <= lmax
    # -lmax <= l = n <= lmax
    for k in range(-lmax, lmax+1):
        l = k
        if 2 * common_functions.omega_list[abs(k)] <= e_max and 2 * common_functions.omega_list[abs(l)] <= e_max:
            op = [k, k, l, l]
            op_list.append([op, 1])
    return op_list
def raise2_op_list(lmax, e_max, m, r):
    op_list = []
    # -lmax <= i < j <= lmax
    for i in range(-lmax, 0):
        if 2 * common_functions.omega_list[abs(i)] <= e_max:
            op = [i, -i]
            op_list.append([op, 2])
    op = [0, 0]
    op_list.append([op, 1])
    return op_list

def raise11_op_list(lmax):
    op_list = []
    # -lmax <= i <= lmax
    for i in range(-lmax, lmax+1):
        op = [i, i]
        op_list.append([op, 1])
    return op_list
def term_4low(lmax, e_max, m, r, basis):
    length_basis=len(basis)
    total_op = csr_matrix(([], ([], [])), shape=(length_basis, length_basis))
    op_list = raise4_op_list(lmax, e_max, m, r)
    for op in op_list:
        i = op[0][0]
        j = op[0][1]
        l = op[0][2]
        k = op[0][3]
        new_op = lower4(int(abs(k)), np.sign(k), int(abs(l)), np.sign(l), int(abs(j)), np.sign(j), int(abs(i)), np.sign(i), e_max, m, r, basis)
        if new_op.count_nonzero() != 0:
            factor = op[1]/math.sqrt(common_functions.omega_list[abs(i)] * common_functions.omega_list[abs(j)] * common_functions.omega_list[abs(l)] * common_functions.omega_list[abs(k)])
            total_op = total_op+factor*new_op
    return total_op

def term_1raise3low(lmax, e_max, m, r, basis):
    
    length_basis = len(basis)
    total_op = csr_matrix(([], ([], [])), shape=(length_basis, length_basis))
    op_list = raise31_op_list(lmax, e_max, m, r)
    for op in op_list:
        i = op[0][0]
        j = op[0][1]
        l = op[0][2]
        # k is the raising operator
        k = op[0][3]
        new_op = raise1low3(int(abs(i)), np.sign(i), int(abs(j)), np.sign(j), int(abs(l)), np.sign(l), int(abs(k)), np.sign(k), e_max, m, r, basis)
        if new_op.count_nonzero() != 0:
            factor = 4 * op[1] / math.sqrt(common_functions.omega_list[abs(i)] * common_functions.omega_list[abs(j)] * common_functions.omega_list[abs(l)] * common_functions.omega_list[abs(k)])
            total_op = total_op + factor * new_op
    return total_op

def term_2raise2low(lmax, e_max, m, r, basis):
    length_basis=len(basis)
    total_op = csr_matrix(([], ([], [])), shape=(length_basis, length_basis))
    op_list = raise22_op_list(lmax, e_max, m, r)
    for op in op_list:
        i = op[0][0]
        j = op[0][1]
        l = op[0][2]
        k = op[0][3]
        new_op = raise2low2(abs(k), np.sign(k), abs(l), np.sign(l), abs(j), np.sign(j), abs(i), np.sign(i), e_max, m, r, basis)
        if new_op.count_nonzero() != 0:
            factor = 6 * op[1] / math.sqrt(common_functions.omega_list[abs(i)] * common_functions.omega_list[abs(j)] * common_functions.omega_list[abs(l)] * common_functions.omega_list[abs(k)])
            total_op = total_op + factor * new_op
    return total_op

def term_2low(lmax, e_max, m, r, basis):
    length_basis = len(basis)
    total_op = csr_matrix(([], ([], [])), shape=(length_basis, length_basis))
    op_list = raise2_op_list(lmax, e_max, m, r)
    for op in op_list:
        i = op[0][0]
        j = op[0][1]
        new_op = lower2(abs(j), np.sign(j), abs(i), np.sign(i), e_max, m, r, basis)
        if new_op.count_nonzero() != 0:
            factor = op[1]/math.sqrt(common_functions.omega_list[abs(i)] * common_functions.omega_list[abs(j)])
            total_op = total_op+factor*new_op
    return total_op

def term_1raise1low(lmax, e_max, m, r, basis):
    row = []
    col = []
    data = []
    length_basis=len(basis)
    for state_index in range(length_basis):
        state = basis[state_index]
        row.append(state_index)
        col.append(state_index)
        factor = 0
        for i in range(len(state)):
            factor = factor + 2 * state[i] / common_functions.omega_list[abs(common_functions.the_l(i))]
        data.append(factor)
    return csr_matrix((data, (row, col)), shape=(length_basis, length_basis))
