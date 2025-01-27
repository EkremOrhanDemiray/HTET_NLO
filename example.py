import common_functions
import genham
import counter_terms
import bubble_operator
import math
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse import identity
import scipy.sparse.linalg as sparse_la
import time
import os
import os.path
import matplotlib.pyplot as plt
from IPython.display import display
from numba import jit
from numba.extending import overload

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

preset_colors=mcolors.cnames
color_names_all = sorted(preset_colors.keys())
i=0
bad_colors=[0, 1, 2, 3, 4, 5, 6, 8, 12, 14, 18, 46, 20, 24, 26, 27, 49, 50, 55, 57, 61, 62, 63, 64, 65, 66,67, 68, 69, 70, 71,72,73, 80, 81, 82, 84,93, 97, 98, 99, 100, 102, 108,109, 110,112,113,115,117,127,129, 134,135,139,143, 144, 145, 146]

    
color_names= color_names_all[32:33]+ color_names_all[120:121]
color_names= color_names +color_names_all[106:107]+color_names_all[105:106]+color_names_all[51:52]
color_names= color_names + color_names_all[147:148]+color_names_all[54:55]+color_names_all[138:139]+color_names_all[21:22]
color_names= color_names +color_names_all[60:61]+color_names_all[118:119]+color_names_all[95:96]+color_names_all[40:41]
color_names= color_names +color_names_all[53:54]+color_names_all[7:8]

m=1
r=20/(2*math.pi)
g=4*math.pi*2

e_max_max=10
l_uv=100

n_eigens_max=10
n_max=100


retVec = []
directory = os.getcwd() + "\\hamiltonians"

if ( os.path.isdir(directory) == True ) :
    print("file path exsist")
    currentDir = directory
else:
    currentDir = os.mkdir(directory)


for e_max in range(9, e_max_max+1):
    
    plt.close()
    lmax=e_max
    print_time=time.time()
    if pow(m, 2) < pow(e_max, 2) / 4:
        lmaxeff = int(min(lmax, math.floor(math.sqrt(max(0, pow(r, 2) * (float(pow(e_max, 2)) / 4. - pow(m, 2)))))))
    else:
        lmaxeff = 0
    n_max=100
    if n_max > int(np.floor(e_max / common_functions.omega(0, m, r))): 
        n_max = int(np.floor(e_max / common_functions.omega(0, m, r)))
    lmax=lmaxeff

    basis=common_functions.basis_l0(common_functions.make_basis(lmax, e_max, m, r))
    basis.sort()
    common_functions.gen_omega_list(l_uv+1, m, r)
    length_basis=len(basis)
    
    n_eigens=n_eigens_max
    
    if n_eigens > length_basis: 
        n_eigens = length_basis
        
    currentDir = os.getcwd() + "\\hamiltonians"
    
    nameOfTheH0 = "H0"+"_"+"{}".format(lmaxeff)+ "_" + "{}".format(e_max)+ "_"+ "{}".format(m) + "_" + "r-20"
    nameOfTheH2 = "H2"+"_"+"{}".format(lmaxeff)+ "_" + "{}".format(e_max)+ "_"+ "{}".format(m) + "_" + "r-20"
    nameOfTheH4 = "H4"+"_"+"{}".format(lmaxeff)+ "_" + "{}".format(e_max)+ "_"+ "{}".format(m) + "_" + "r-20"
    
    dirOfH0 = currentDir + "\\" + nameOfTheH0
    dirOfH2 = currentDir + "\\" + nameOfTheH2
    dirOfH4 = currentDir + "\\" + nameOfTheH4
    
    if (os.path.isfile(dirOfH0+".npz") == True ):

        h0Mat = sparse.load_npz("{}.npz".format(dirOfH0))
        
    else:
        
        h0Mat = genham.h0(lmaxeff, e_max, m, r, basis)
        sparse.save_npz("{}".format(dirOfH0),h0Mat)
        
        
    if (os.path.isfile(dirOfH2+".npz") == True ):
        h2Mat = sparse.load_npz("{}.npz".format(dirOfH2))
        
    else:
        h2Mat = genham.delta_h2(lmaxeff, e_max, m, r, basis)
        sparse.save_npz("{}".format(dirOfH2),h2Mat)
        
        
    if (os.path.isfile(dirOfH4+".npz") == True ):
        h4Mat = sparse.load_npz("{}.npz".format(dirOfH4))
        
    else:
        h4Mat = genham.delta_h4(lmaxeff, e_max, m, r, basis)
        sparse.save_npz("{}".format(dirOfH4),h4Mat)
        
    h2Mat=1./4.*h2Mat
    h4Mat=1/r*1/(8*math.pi)*h4Mat
    
    hRaw = h0Mat + 1/24* g*h4Mat
    
    hLO = h0Mat + counter_terms.mv2_sq(e_max, m, r, g, l_uv)*h2Mat + 1/24* (g + counter_terms.lambda2(e_max, m, r, g, l_uv))*h4Mat
    
    hNLO = hLO + 0.5*counter_terms.beta2(e_max,m,r,g,l_uv)*(h0Mat.multiply(h2Mat)-h2Mat.multiply(h0Mat))
    hNLO += 0.5*counter_terms.beta1(e_max,m,r,g,l_uv)*h0Mat.multiply(h2Mat) + bubble_operator.bubbleOperator(m,r,basis,e_max,l_uv,g)
    hNLO += (1/24)*counter_terms.alpha1(e_max,m,r,g,l_uv)*(h4Mat.multiply(h0Mat) - h0Mat.multiply(h4Mat))
    hNLO += (1/24)*counter_terms.alpha2(e_max,m,g)*h0Mat.multiply(h4Mat)
    
    

    if n_eigens > length_basis-2:
        
        NLONorm = scipy.sparse.linalg.norm(hNLO)
        eigensOfNLO = sparse_la.eigs(hNLO - NLONorm * identity(length_basis), n_eigens, None, None, which='LM',)[0]
        eigensOfNLO = np.array(eigensOfNLO.real) + NLONorm

        output_array = [[l_uv, e_max]]
        
        output_array.append(sorted(eigensOfNLO.real)[:n_eigens])
        
    else:
        
        NLONorm = scipy.sparse.linalg.norm(hNLO)
        eigensOfNLO = sparse_la.eigs(hNLO - NLONorm * identity(length_basis), n_eigens, None, None, which='LM',)[0]
        eigensOfNLO = np.array(eigensOfNLO.real) + NLONorm
        
        output_array = [[l_uv, e_max]]
        
        output_array.append(sorted(eigensOfNLO.real)[:n_eigens])
    retVec.append(output_array)

    
    print(e_max)
    color_i=0
    current_figure_objects=[]

emax_data=[x[0][1] for x in retVec]
emax_data3=[pow(x[0][1],-3) for x in retVec]
emax_data4=[pow(x[0][1],-4) for x in retVec]

e10_dataNLO=[x[1][1]-x[1][0] if len(x[1])>1 else None for x in retVec]

i = 0
if len(output_array)!=0:
    plt.plot(emax_data, e10_dataNLO, marker="o", color=color_names[color_i -2 % len(color_names)], label="NLO")
i+= 1

plt.figure(1)
plt.title("l_uv={}".format(retVec[0][0][0]),loc="left")
plt.title("g = 8*pi_emax", loc = "center")
plt.title("r = 20/2pi",loc = "right")
plt.xlabel("E_max")
plt.ylabel("E_n")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
display(plt.gcf()) 

plt.figure(2)
if len(output_array)!=0:
    plt.plot(emax_data3, e10_dataNLO, marker="o", color=color_names[color_i -2 % len(color_names)], label="NLO")
plt.figure(2)
plt.title("l_uv={}".format(retVec[0][0][0]),loc="left")
plt.title("g = 8*pi_e^-3", loc = "center")
plt.title("r = 20/2pi",loc = "right")
plt.xlabel("E_max^-3")
plt.ylabel("E_n")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
display(plt.gcf()) 

plt.figure(3)

if len(output_array)!=0:
    plt.plot(emax_data4, e10_dataNLO, marker="o", color=color_names[color_i -2 % len(color_names)], label="NLO")
plt.figure(3)
plt.title("l_uv={}".format(retVec[0][0][0]),loc="left")
plt.title("g = 8*pi_e^-4", loc = "center")
plt.title("r = 20/2pi",loc = "right")
plt.xlabel("E_max^-4")
plt.ylabel("E_n")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
display(plt.gcf())
