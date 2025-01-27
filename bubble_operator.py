import math
from scipy.sparse import csr_matrix
import common_functions
from numba import jit
from numba.extending import overload

@jit(nopython=True)
def calTermsForBubble(m,r, finalEnergy,maxEnergy,angularMomUV):
    omegaListJit = []
    for i in range(angularMomUV+1):
       omegaListJit.append(math.sqrt(float(i) * float(i) / (r*r) + m*m))
    
    sumTerm1 = 0
    sumTerm2 = 0
    sumTerm3 = 0
    sumTerm4 = 0
    sumTerm5 = 0
    sumTerm6 = 0
    sumTerm7 = 0
    sumTerm8 = 0
    sumTerm9  = 0
    sumTerm10  = 0
    sumTerm11  = 0
    sumTerm12  = 0
    sumTerm13  = 0
    sumTerm14  = 0
    sumTerm15  = 0
    sumTerm16 = 0
    retVal = 0 
    
    if ( finalEnergy + 4*m - maxEnergy > 0):
        sumTerm1 = 1/(-4*pow(m,5))   
        
    for i in range(1,angularMomUV+1):
        if ( finalEnergy + 2*m + 2*omegaListJit[i] - maxEnergy>0):
            sumTerm2 -= 1/(pow(omegaListJit[i],2)*(m+omegaListJit[i]))
        if ( finalEnergy + 4*omegaListJit[i]- maxEnergy > 0):
            sumTerm4 -= 1/(pow(omegaListJit[i],5))

            
    sumTerm2 *= 6/pow(m,2)
    sumTerm4 *= (3/2)
    
    for i in range(1, math.floor(angularMomUV/2) + 1):
        if ( finalEnergy + m + 2*omegaListJit[i] + omegaListJit[2*i]-maxEnergy>0):
            sumTerm3 -= 1/(pow(omegaListJit[i],2)*omegaListJit[2*i]*(m + 2*omegaListJit[i] + omegaListJit[2*i]))

            
    sumTerm3*= 6/m
    
    for i in range(1,math.floor(angularMomUV/3) + 1 ):
        if ( finalEnergy + 3*omegaListJit[i] + omegaListJit[3*i] - maxEnergy >0):
            sumTerm5 -= 1/(pow(omegaListJit[i],3) * omegaListJit[3*i] * ( 3*omegaListJit[i] + omegaListJit[3*i]))
  
    sumTerm5*= 2
    
    for i in range(1,math.floor((angularMomUV-1)/2) + 1 ):
        for j in range(i+1, angularMomUV - i + 1):
            if ( finalEnergy + m + omegaListJit[i] + omegaListJit[j] + omegaListJit[i + j] - maxEnergy >0):
                num = 1
                denom = omegaListJit[i]*omegaListJit[j]*omegaListJit[i+j]*(m+omegaListJit[i]+omegaListJit[j]+omegaListJit[i+j])
                sumTerm6 -= (num/denom)
     
    sumTerm6 *= 12/m
    
    for i in range(1,angularMomUV):
        for j in range(i+1,angularMomUV+1):
            if ( finalEnergy + m + omegaListJit[i] + omegaListJit[j] +omegaListJit[abs(i-j)] - maxEnergy > 0):
                sumTerm7 -= 1/(omegaListJit[i]*omegaListJit[j]*omegaListJit[abs(i-j)]*(m+omegaListJit[i] + omegaListJit[j] +omegaListJit[abs(i-j)]))

            if ( finalEnergy + 2*omegaListJit[i] + 2*omegaListJit[j] - maxEnergy > 0):
                sumTerm8 -= 1/(pow(omegaListJit[i],2)*pow(omegaListJit[j],2)*(omegaListJit[i] + omegaListJit[j]))

            if ( finalEnergy + 2*omegaListJit[i] + omegaListJit[j] + omegaListJit[abs(2*i-j)]-maxEnergy > 0):
                sumTerm11 -= 1/(pow(omegaListJit[i],2)*omegaListJit[j]*omegaListJit[abs(2*i-j)]*(2*omegaListJit[i]+omegaListJit[j]+ omegaListJit[abs(2*i-j)]))

    
    sumTerm7 *= 12/m
    sumTerm8 *= 12
    sumTerm11 *= 6
    
    for i in range(1,math.floor((angularMomUV-1)/3) + 1):
        for j in range(i+1,angularMomUV-2*i + 1):
            if ( finalEnergy + 2*omegaListJit[i] + omegaListJit[j] + omegaListJit[2*i+j]-maxEnergy > 0):
                sumTerm9 -= 1/(pow(omegaListJit[i],2)*omegaListJit[j]*omegaListJit[2*i+j]*(2*omegaListJit[i]+omegaListJit[j]+ omegaListJit[2*i+j]))

    sumTerm9*= 6
    
    for i in range(1,math.floor((angularMomUV-2)/3) + 1):
        for j in range(i+1,math.floor((angularMomUV-i)/2) + 1):
            if ( finalEnergy + omegaListJit[i] + 2*omegaListJit[j] + omegaListJit[i+2*j] - maxEnergy > 0):
                sumTerm10 -= 1/(omegaListJit[i]*pow(omegaListJit[j],2) * omegaListJit[i+2*j]*(omegaListJit[i] + 2*omegaListJit[j] + omegaListJit[i+2*j]))

    sumTerm10*= 6
    
    for i in range(1,angularMomUV-1):
        for j in range(i+1,math.floor((angularMomUV+i)/2) + 1):
            if ( finalEnergy + omegaListJit[i] + 2*omegaListJit[j] + omegaListJit[abs(2*j-i)] - maxEnergy > 0):
                sumTerm12 -= 1 / ( omegaListJit[i]*pow(omegaListJit[j],2)*omegaListJit[abs(2*j-i)]*(omegaListJit[i] + 2*omegaListJit[j] + omegaListJit[abs(2*j-i)]))    

    sumTerm12*= 6
    
    for i in range(1,math.floor((angularMomUV-3)/3)+1):
        for j in range(i+1,math.floor((angularMomUV-i-1)/2) + 1):
            for k in range(j+1,angularMomUV -i-j + 1):
                if ( finalEnergy + omegaListJit[i] + omegaListJit[j] + omegaListJit[k] + omegaListJit[i+j+k] - maxEnergy>0):
                    sumTerm13 -= 1/(omegaListJit[i]*omegaListJit[j]*omegaListJit[k]*omegaListJit[abs(i+j+k)]*(omegaListJit[i] + omegaListJit[j] + omegaListJit[k] + omegaListJit[abs(i+j+k)]))

    sumTerm13 *= 12
    
    for i in range(1,angularMomUV-1):
        for j in range(i+1,angularMomUV):
            for k in range(j+1,angularMomUV+1):
                if ( finalEnergy + omegaListJit[i] + omegaListJit[j] + omegaListJit[k] + omegaListJit[abs(i+j-k)] - maxEnergy > 0):
                    sumTerm14 -= 1/(omegaListJit[i]*omegaListJit[j]*omegaListJit[k]*omegaListJit[abs(i+j-k)]*(omegaListJit[i]+omegaListJit[j]+omegaListJit[k]+omegaListJit[abs(i+j-k)]))

                if ( finalEnergy + omegaListJit[i] + omegaListJit[j] + omegaListJit[k] + omegaListJit[abs(i-j+k)]-maxEnergy > 0):
                    sumTerm15 -= 1/(omegaListJit[i]* omegaListJit[j]*omegaListJit[k]* omegaListJit[abs(i-j+k)]*(omegaListJit[i] + omegaListJit[j] + omegaListJit[k] + omegaListJit[abs(i-j+k)]))


    sumTerm14*=12
    sumTerm15*=12
    
    for i in range(1,angularMomUV-2):
        for j in range(i+1, math.floor((angularMomUV+i-1)/2) + 1):
            for k in range(j+1,angularMomUV-j+i + 1):
                if ( finalEnergy + omegaListJit[i] + omegaListJit[j] + omegaListJit[k] + omegaListJit[abs(-i+j+k)]- maxEnergy > 0):
                    sumTerm16 -= 1/(omegaListJit[i]*omegaListJit[j]*omegaListJit[k]*omegaListJit[abs(-i+j+k)]*(omegaListJit[i]+omegaListJit[j]+omegaListJit[k]+omegaListJit[abs(-i+j+k)]))

    sumTerm16*= 12
    retVal = (sumTerm1+sumTerm2 +sumTerm3 + sumTerm4 + sumTerm5 + sumTerm6 + sumTerm7 + sumTerm8 + sumTerm9 + sumTerm10 + sumTerm11 + sumTerm12 + sumTerm13 + sumTerm14 + sumTerm15 + sumTerm16 )
    
    return retVal

def bubbleOperator(m,r,basis,maxEnergy,angularMomUV,g):
    row=[]
    col=[]
    data=[]
    
    for idx in range(len(basis)):
        
        retVal = 0
        
        state = basis[idx]
        finalEnergy = common_functions.state_energy(state, m, r)
        retVal = calTermsForBubble(m,r,finalEnergy,maxEnergy,angularMomUV)
        row.append(idx)
        col.append(idx)
        data.append(1/24*pow(g, 2)/pow(2*math.pi*r, 2)*retVal/16)
    
    return csr_matrix((data, (row, col)), shape=(len(basis), len(basis)))

#after some approximations, we get c1Coeff, which is a scalar and is much faster to compute
@jit(nopython=True)
def c1Coeff(m,r,g,maxEnergy, angularMomUV):
    retVal = 0  
    
    omegaList = []
    sumTerm1 = 0
    sumTerm2 = 0
    sumTerm3 = 0
    sumTerm4 = 0
    sumTerm5 = 0
    sumTerm6 = 0
    sumTerm7 = 0
    sumTerm8 = 0
    sumTerm9  = 0
    sumTerm10  = 0
    for i in range(angularMomUV+1):
       omegaList.append(math.sqrt(float(i) * float(i) / (r*r) + m*m))
    
    for k in range(1,angularMomUV):
        for kPrime in range(k+1,angularMomUV):
            if ( maxEnergy -omegaList[k] - omegaList[kPrime] -k/r - 3*kPrime/r -2/r > 0):
                sumTerm1 += 1/( omegaList[k]*omegaList[kPrime]*(pow(maxEnergy - omegaList[k]-omegaList[kPrime],2) -pow(k+kPrime,2)/pow(r,2) ))
            if ( maxEnergy - omegaList[k] - omegaList[kPrime] -k/r - kPrime/r -2/r > 0):
                sumTerm1 += 1/( omegaList[k]*omegaList[kPrime]*(pow(maxEnergy - omegaList[k]-omegaList[kPrime],2) -pow(k+kPrime,2)/pow(r,2) ))
            if ( maxEnergy -omegaList[k] - omegaList[kPrime] -k/r - kPrime/r -2/r > 0):
                sumTerm2 += 1/( omegaList[k]*omegaList[kPrime]*(pow(maxEnergy - omegaList[k]-omegaList[kPrime],2) -pow(k-kPrime,2)/pow(r,2) ))
            if ( maxEnergy - omegaList[k] - omegaList[kPrime] + k/r - 3*kPrime/r -2/r > 0):
                sumTerm2 += 1/( omegaList[k]*omegaList[kPrime]*(pow(maxEnergy - omegaList[k]-omegaList[kPrime],2) -pow(k-kPrime,2)/pow(r,2) ))
                
    for k in range(1,angularMomUV):
        for kPrime in range(1,k-1):
            if ( maxEnergy - omegaList[abs(k-kPrime)] -omegaList[k] - kPrime/r - 2*k/r - 2/r > 0):
                sumTerm3 += 1/(omegaList[k]*omegaList[abs(k-kPrime)]*(pow(maxEnergy - omegaList[abs(k-kPrime)] -omegaList[k],2) -pow(kPrime,2)/pow(r,2)))
    
    
    for k in range(1,angularMomUV): 
        if ( (maxEnergy - omegaList[k] - k/r -2/r) > 0 and (-maxEnergy + omegaList[k] + 3*k/r -4/r) > 0):
            sumTerm4 += 2/(omegaList[k]*(maxEnergy - omegaList[k] + k/r)*(pow(maxEnergy - omegaList[k],2) - pow(k,2)/pow(r,2)))
            
    for  k in range(1,angularMomUV):
        if ( maxEnergy - 2*omegaList[k]-4*k/r -2/r >0):
            sumTerm5 += 0.5/(pow(omegaList[k],2)*(pow(maxEnergy-2*omegaList[k],2)-4*pow(k,2)/pow(r,2)))
        if ( maxEnergy - 2*omegaList[k]-2*k/r-2/r > 0):
            sumTerm5 += 0.5/(pow(omegaList[k],2)*(pow(maxEnergy-2*omegaList[k],2)-4*pow(k,2)/pow(r,2)))
    
    for k in range(1,angularMomUV):
        if ( maxEnergy - omegaList[k] -5*k/r - 4/r > 0):
            sumTerm6 += 2/(omegaList[k]*(maxEnergy-omegaList[k] - k/r )*(pow(maxEnergy-omegaList[k],2)-pow(k,2)/pow(r,2)))
    for k in range(1,angularMomUV):
        if ( maxEnergy -omegaList[k] -3*k/r - 4/r > 0):
            sumTerm7 += 2/(omegaList[k]*(maxEnergy-omegaList[k]+k/r)*(pow(maxEnergy - omegaList[k],2)-pow(k,2)/pow(r,2))) 
    for k in range(1,angularMomUV):
        if ( maxEnergy - 2*omegaList[k] -2*k/r - 2/r > 0):
            sumTerm8 += 2/(pow(omegaList[k],2)*pow(maxEnergy - 2*omegaList[k],2))
    for k in range(1,angularMomUV):
        if ( maxEnergy - m - omegaList[k] - 3*k/r -2/r > 0):
            sumTerm9 += 2/(omegaList[k]*m*(pow(maxEnergy-m-omegaList[k],2)-pow(k,2)/pow(r,2)))
        if ( maxEnergy - m - omegaList[k] - k/r - 2/r > 0):
            sumTerm9 += 1/(omegaList[k] * m * ( pow(maxEnergy - m - omegaList[k],2 ) - pow(k,2)/pow(r,2)))
    sumTerm10 += 1/(pow(maxEnergy-2*m,2)*pow(m,2))
    
    retVal = sumTerm1 + sumTerm2 + sumTerm3 + sumTerm4 + sumTerm5 + sumTerm6 + sumTerm7 + sumTerm8 + sumTerm9 + sumTerm10 
    
    retVal = -r*pow(g,2)/(16*maxEnergy*pow(2*math.pi*r,2))*retVal
    
    return retVal