#True Version 
from scipy.sparse.linalg import eigs
import common_functions
import genham
import counter_terms
import bubble_operator

m=1
r=20/(2*math.pi)
gg = np.linspace(0,10*math.pi*4, 25)
gg = gg[3:]
e_max_max = [18]

l_uv=1000

n_eigens_max=50
basis_name=""
n_max=100
saveDir = r"C:\\Users\\EkremDemiray\Desktop\\HamiltonianTruncation\\HamiltonianTruncation\\HamiltonianTruncation\\criticalCouplingHopefullyFinalVersion//"
k = 0
finalVec = []
for g in gg:
    for e_max in e_max_max:
        
        retVec = []

        lmax=e_max
        print_time=time.time()
        if pow(m, 2) < pow(e_max, 2) / 4:
            lmaxeff = int(min(lmax, math.floor(math.sqrt(max(0, pow(r, 2) * (float(pow(e_max, 2)) / 4. - pow(m, 2)))))))
        else:
            lmaxeff = 0
        n_max=100
        if n_max > int(np.floor(e_max / omega(0, m, r))): 
            n_max = int(np.floor(e_max / omega(0, m, r)))
        lmax=lmaxeff

            
        basis=basis_even(basis_l0(make_basis(lmax, e_max, m, r)))
        basis.sort()

        basisOdd=basis_odd(basis_l0(make_basis(lmax, e_max, m, r)))
        basisOdd.sort()
        x = time.time()
        dirOfBasisOdd = os.getcwd() + "\\oddBasisHamiltonian"+"\\" + "{}".format(lmax)+ "_" + "{}".format(e_max) + "{}".format(r) + "_Odd.txt"

        if ( os.path.isfile(dirOfBasisOdd) == True):
            basisOdd = np.loadtxt(dirOfBasisOdd).astype(np.int64)
            
        else:
            basisOdd=basis_odd(basis_l0(make_basis(lmax, e_max, m, r)))
            basisOdd.sort()
            basisOddMatCast = np.matrix(basisOdd)
            with open(dirOfBasisOdd, 'wb') as f:
                for line in basisOddMatCast:
                    np.savetxt(f, line)
                    
        
        
        dirOfBasisEven = os.getcwd() + "\\evenBasisHamiltonian"+"\\" + "{}".format(lmax)+ "_" + "{}".format(e_max) + "{}".format(r) + "_Even.txt"
        if ( os.path.isfile(dirOfBasisEven) == True):
            basis = np.loadtxt(dirOfBasisEven).astype(np.int64)
            
        else:
            basis=basis_even(basis_l0(make_basis(lmax, e_max, m, r)))
            basis.sort()
            basisEvenMatCast = np.matrix(basis)
            with open(dirOfBasisEven, 'wb') as f:
                for line in basisEvenMatCast:
                    np.savetxt(f, line)
                    
        gen_omega_list(l_uv+1, m, r)
        
        length_basis=len(basis)
        
        length_basisOdd=len(basisOdd)
        

        n_eigens=n_eigens_max

        if n_eigens > length_basis: 
            n_eigens = length_basis


        currentDirOdd = os.getcwd() + "\\oddBasisHamiltonian"

        nameOfTheH0Odd = "H0"+"_"+"{}".format(lmaxeff)+ "_" + "{}".format(e_max)+ "_"+ "{}".format(m) + "_" + "r-20_Odd"
        nameOfTheH2Odd = "H2"+"_"+"{}".format(lmaxeff)+ "_" + "{}".format(e_max)+ "_"+ "{}".format(m) + "_" + "r-20_Odd"
        nameOfTheH4Odd = "H4"+"_"+"{}".format(lmaxeff)+ "_" + "{}".format(e_max)+ "_"+ "{}".format(m) + "_" + "r-20_Odd"

        dirOfH0Odd = currentDirOdd + "\\" + nameOfTheH0Odd
        dirOfH2Odd = currentDirOdd + "\\" + nameOfTheH2Odd
        dirOfH4Odd = currentDirOdd + "\\" + nameOfTheH4Odd

        if (os.path.isfile(dirOfH0Odd+".npz") == True ):
            h0MatOdd = sparse.load_npz("{}.npz".format(dirOfH0Odd))
            
        else:
            h0MatOdd = h0(lmaxeff, e_max, m, r, basisOdd)
            sparse.save_npz("{}".format(dirOfH0Odd),h0MatOdd)
            
            
        if (os.path.isfile(dirOfH2Odd+".npz") == True ):
            h2MatOdd = sparse.load_npz("{}.npz".format(dirOfH2Odd))
            
        else:
            h2MatOdd = delta_h2(lmaxeff, e_max, m, r, basisOdd)
            sparse.save_npz("{}".format(dirOfH2Odd),h2MatOdd)
            
            
        if (os.path.isfile(dirOfH4Odd+".npz") == True ):
            h4MatOdd = sparse.load_npz("{}.npz".format(dirOfH4Odd))
            
        else:
            h4MatOdd = delta_h4(lmaxeff, e_max, m, r, basisOdd)
            sparse.save_npz("{}".format(dirOfH4Odd),h4MatOdd)

        h2MatOdd=1./4.*h2MatOdd
        h4MatOdd=1/r*1/(8*math.pi)*h4MatOdd

        hRawOdd = h0MatOdd + 1/24* g*h4MatOdd

        hLOOdd = h0MatOdd + mv2_sqEkrem(e_max, m, r, g, l_uv)*h2MatOdd + 1/24* (g + lambda2Ekrem(e_max, m, r, g, l_uv))*h4MatOdd

        hNLOOdd = (1/24)*counter_terms.alpha1(e_max,m,r,g,l_uv)*(np.dot(h4MatOdd,h0MatOdd) - np.dot(h0MatOdd,h4MatOdd)) + (1/24)*counter_terms.alpha2(e_max,m,g)*np.dot(h0MatOdd,h4MatOdd)
        hNLOOdd += 0.5*counter_terms.beta1(e_max,m,r,g,l_uv)*np.dot(h0MatOdd,h2MatOdd)+ 0.5*counter_terms.beta2(e_max,m,r,g,l_uv)*(np.dot(h0MatOdd,h2MatOdd) -np.dot(h2MatOdd,h0MatOdd))
        hNLOOdd +=  hLOOdd + (bubble_operator.c1Coeff(m,r,g,e_max,l_uv) )*h0MatOdd + bubble_operator.bubbleOperatorConstantTerm(m,r,e_max,l_uv,g)*h0MatOdd

        #even start
        currentDirEven = os.getcwd() + "\\evenBasisHamiltonian"
        
        nameOfTheH0 = "H0"+"_"+"{}".format(lmaxeff)+ "_" + "{}".format(e_max)+ "_"+ "{}".format(m) + "_" + "r-20_even"
        nameOfTheH2 = "H2"+"_"+"{}".format(lmaxeff)+ "_" + "{}".format(e_max)+ "_"+ "{}".format(m) + "_" + "r-20_even"
        nameOfTheH4 = "H4"+"_"+"{}".format(lmaxeff)+ "_" + "{}".format(e_max)+ "_"+ "{}".format(m) + "_" + "r-20_even"

        dirOfH0 = currentDirEven + "\\" + nameOfTheH0
        dirOfH2 = currentDirEven + "\\" + nameOfTheH2
        dirOfH4 = currentDirEven + "\\" + nameOfTheH4


        if (os.path.isfile(dirOfH0+".npz") == True ):
            h0Mat = sparse.load_npz("{}.npz".format(dirOfH0))
            
        else:
            h0Mat = h0(lmaxeff, e_max, m, r, basis)
            sparse.save_npz("{}".format(dirOfH0),h0Mat)
            
            
        if (os.path.isfile(dirOfH2+".npz") == True ):
            h2Mat = sparse.load_npz("{}.npz".format(dirOfH2))
            
        else:
            h2Mat = delta_h2(lmaxeff, e_max, m, r, basis)
            sparse.save_npz("{}".format(dirOfH2),h2Mat)
            
            
        if (os.path.isfile(dirOfH4+".npz") == True ):
            h4Mat = sparse.load_npz("{}.npz".format(dirOfH4))
            
        else:
            h4Mat = delta_h4(lmaxeff, e_max, m, r, basis)
            sparse.save_npz("{}".format(dirOfH4),h4Mat)

        h2Mat=1./4.*h2Mat
        h4Mat=1/r*1/(8*math.pi)*h4Mat

        hRaw = h0Mat + 1/24* g*h4Mat

        hLO = h0Mat + mv2_sqEkrem(e_max, m, r, g, l_uv)*h2Mat + 1/24* (g + lambda2Ekrem(e_max, m, r, g, l_uv))*h4Mat

        hNLO = (1/24)*counter_terms.alpha1(e_max,m,r,g,l_uv)*(np.dot(h4Mat,h0Mat) - np.dot(h0Mat,h4Mat)) + (1/24)*counter_terms.alpha2(e_max,m,g)*np.dot(h0Mat,h4Mat)
        hNLO += 0.5*counter_terms.beta1(e_max,m,r,g,l_uv)*np.dot(h0Mat,h2Mat)+ 0.5*counter_terms.beta2(e_max,m,r,g,l_uv)*(np.dot(h0Mat,h2Mat) -np.dot(h2Mat,h0Mat))
        hNLO +=  counter_terms.mv2_sq(e_max, m, r, g, l_uv)*h2Mat + 1/24* (g + counter_terms.lambda2(e_max, m, r, g, l_uv))*h4Mat + h0Mat + (bubble_operator.c1Coeff(m,r,g,e_max,l_uv) )*h0Mat + bubble_operator.bubbleOperatorConstantTerm(m,r,e_max,l_uv,g)*h0Mat
        
        y = time.time()
        
        print("time for basis and hamiltonian generation: ", y-x, "for g: ", g)
        
        x = time.time()
        if n_eigens > length_basis-2:



            hs_normOddRaw=scipy.sparse.linalg.norm(hRawOdd)
            eigensOddRaw = sparse_la.eigs(hRawOdd - hs_normOddRaw * identity(length_basisOdd), n_eigens, None, None, which='LM',)[0] ##faster line
            eigensOddRaw = np.array(eigensOddRaw.real) + hs_normOddRaw
            
            
            hs_normEvenRaw=scipy.sparse.linalg.norm(hRaw)
            eigensEvenRaw = sparse_la.eigs(hRaw - hs_normEvenRaw * identity(length_basis), n_eigens, None, None, which='LM',)[0] ##faster line
            eigensEvenRaw = np.array(eigensEvenRaw.real) + hs_normEvenRaw
            
            hs_normOddLO=scipy.sparse.linalg.norm(hLOOdd)
            eigensOddLO = sparse_la.eigs(hLOOdd - hs_normOddLO * identity(length_basisOdd), n_eigens, None, None, which='LM',)[0] ##faster line
            eigensOddLO = np.array(eigensOddLO.real) + hs_normOddLO
            
            
            hs_normEvenLO=scipy.sparse.linalg.norm(hLO)
            eigensEvenLO = sparse_la.eigs(hLO - hs_normEvenLO * identity(length_basis), n_eigens, None, None, which='LM',)[0] ##faster line
            eigensEvenLO = np.array(eigensEvenLO.real) + hs_normEvenLO
            
            
            
            hs_normOdd=scipy.sparse.linalg.norm(hNLOOdd)
            eigensOdd = sparse_la.eigs(hNLOOdd - hs_normOdd * identity(length_basisOdd), n_eigens, None, None, which='LM',)[0] ##faster line
            eigensOdd = np.array(eigensOdd.real) + hs_normOdd
            
            
            hs_normEven=scipy.sparse.linalg.norm(hNLO)
            eigensEven = sparse_la.eigs(hNLO - hs_normEven * identity(length_basis), n_eigens, None, None, which='LM',)[0] ##faster line
            eigensEven = np.array(eigensEven.real) + hs_normEven
            
        
            # eigensOfNLOEven = scipy.linalg.eigvals((hNLO).toarray())
            # eigensOfNLOOdd = scipy.linalg.eigvals((hNLOOdd).toarray())
            
            output_array = [[l_uv, e_max]]
            
            output_array.append(sorted(eigensEven.real)[:n_eigens])
            output_array.append(sorted(eigensOdd.real)[:n_eigens])
            
            output_array.append(sorted(eigensEvenLO.real)[:n_eigens])
            output_array.append(sorted(eigensOddLO.real)[:n_eigens])
            
            output_array.append(sorted(eigensEvenRaw.real)[:n_eigens])
            output_array.append(sorted(eigensOddRaw.real)[:n_eigens])
            
            
            
            # output_array.append(sorted(eigens.real)[:n_eigens])
            # output_array.append(sorted(eigensOfRaw.real)[:n_eigens])
            
        else:
            
            hs_normOddRaw=scipy.sparse.linalg.norm(hRawOdd)
            eigensOddRaw = sparse_la.eigs(hRawOdd - hs_normOddRaw * identity(length_basisOdd), n_eigens, None, None, which='LM',)[0] ##faster line
            eigensOddRaw = np.array(eigensOddRaw.real) + hs_normOddRaw
            
            
            hs_normEvenRaw=scipy.sparse.linalg.norm(hRaw)
            eigensEvenRaw = sparse_la.eigs(hRaw - hs_normEvenRaw * identity(length_basis), n_eigens, None, None, which='LM',)[0] ##faster line
            eigensEvenRaw = np.array(eigensEvenRaw.real) + hs_normEvenRaw
            
            hs_normOddLO=scipy.sparse.linalg.norm(hLOOdd)
            eigensOddLO = sparse_la.eigs(hLOOdd - hs_normOddLO * identity(length_basisOdd), n_eigens, None, None, which='LM',)[0] ##faster line
            eigensOddLO = np.array(eigensOddLO.real) + hs_normOddLO
            
            
            hs_normEvenLO=scipy.sparse.linalg.norm(hLO)
            eigensEvenLO = sparse_la.eigs(hLO - hs_normEvenLO * identity(length_basis), n_eigens, None, None, which='LM',)[0] ##faster line
            eigensEvenLO = np.array(eigensEvenLO.real) + hs_normEvenLO
            
            
            
            hs_normOdd=scipy.sparse.linalg.norm(hNLOOdd)
            eigensOdd = sparse_la.eigs(hNLOOdd - hs_normOdd * identity(length_basisOdd), n_eigens, None, None, which='LM',)[0] ##faster line
            eigensOdd = np.array(eigensOdd.real) + hs_normOdd
            
            
            hs_normEven=scipy.sparse.linalg.norm(hNLO)
            eigensEven = sparse_la.eigs(hNLO - hs_normEven * identity(length_basis), n_eigens, None, None, which='LM',)[0] ##faster line
            eigensEven = np.array(eigensEven.real) + hs_normEven
            
            output_array = [[l_uv, e_max]]
            
            output_array.append(sorted(eigensEven.real)[:n_eigens])
            output_array.append(sorted(eigensOdd.real)[:n_eigens])
            
            output_array.append(sorted(eigensEvenLO.real)[:n_eigens])
            output_array.append(sorted(eigensOddLO.real)[:n_eigens])
            
            output_array.append(sorted(eigensEvenRaw.real)[:n_eigens])
            output_array.append(sorted(eigensOddRaw.real)[:n_eigens])
            
            # output_array.append(sorted(eigens.real)[:n_eigens])
            # output_array.append(sorted(eigensOfRaw.real)[:n_eigens])
        retVec.append(output_array)

        y = time.time()
        
        print("Time for the calculation of the eigens is: {}".format(y-x),"for g: ", g)
        
        e10_dataNLO = float(retVec[0][2][0] - retVec[0][1][0])
        e20_dataNLO = float(retVec[0][1][1] - retVec[0][1][0])
        
        e10_dataLO = float(retVec[0][4][0] - retVec[0][3][0])
        e20_dataLO = float(retVec[0][3][1] - retVec[0][3][0])
        
        e10_dataRaw = float(retVec[0][6][0] - retVec[0][5][0])
        e20_dataRaw = float(retVec[0][5][1] - retVec[0][5][0])
        
        eigensOfEvenRaw = retVec[0][5]
        eigensOfOddRaw = retVec[0][6]
        eigensOfEvenLO = retVec[0][3]
        eigensOfOddLO = retVec[0][4]
        eigensOfEvenNLO = retVec[0][1]
        eigensOfOddNLO = retVec[0][2]
            
        # e10_dataNLO=[float(x[2][0] - x[1][0]) if len(x[1])>1 else None for x in retVec]
        # e20_dataNLO = [float(x[1][1] - x[1][0]) if len(x[1])>1 else None for x in retVec]
        finalVec.append([g,e_max,e10_dataNLO,e20_dataNLO,e10_dataLO,e20_dataLO,e10_dataRaw,e20_dataRaw, ])

    # dataFrameForCriticalCoupling = pd.DataFrame({'g' : [x[0] for x in finalVec],'e_max' : [x[1] for x in finalVec], 'Delta_E1-NLO': [x[2] for x in finalVec], 'Delta_E2-NLO': [x[3] for x in finalVec] , 'Delta_E1-LO' : [x[4] for x in finalVec],'Delta_E2-LO' : [x[5] for x in finalVec],'Delta_E1-Raw' : [x[6] for x in finalVec], 'Delta_E2-Raw' : [x[7] for x in finalVec] })
    # dataToExcelForCriticalCoupling = pd.ExcelWriter(saveDir + "g_{}_deltaE1vsEmax_HighVolume_Scaling.xlsx".format(g))
    # dataFrameForCriticalCoupling.to_excel(dataToExcelForCriticalCoupling)
    # dataToExcelForCriticalCoupling.close()
    
    gVec = np.zeros(len(eigensOfEvenRaw))
    gVec.fill(g)
    dataFrameForCriticalCoupling1 = pd.DataFrame({'g':gVec,'eigensOfEvenRaw' : eigensOfEvenRaw, 'eigensOfOddRaw' : eigensOfOddRaw, 'eigensOfEvenLO' : eigensOfEvenLO, 'eigensOfOddLO' : eigensOfOddLO, 'eigensOfEvenNLO' : eigensOfEvenNLO, 'eigensOfOddNLO' : eigensOfOddNLO})
    dataToExcelForCriticalCoupling1 = pd.ExcelWriter(saveDir + "_g_{}_EnergyEigenvalues.xlsx".format(g))
    dataFrameForCriticalCoupling1.to_excel(dataToExcelForCriticalCoupling1)
    dataToExcelForCriticalCoupling1.close()
