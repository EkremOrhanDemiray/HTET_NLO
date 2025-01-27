import math
import numpy as np
import common_functions

#Counter Terms
def mv2_sq(e_max, m, r, g, l_uv):
    sumTerm1 = 0
    sumTerm2 = 0
    sumTerm3 = 0
    
    for k in range(1, math.floor((l_uv-1)/2)+1):
        for kPrime in range(k+1,l_uv-k+1):
            sumTerm1 += -pow(g,2)/pow((2*math.pi*r),2)*0.5*np.heaviside(common_functions.omega(k,m,r) + common_functions.omega(kPrime,m,r) + common_functions.omega(k+kPrime,m,r)-e_max,0)/(common_functions.omega(k,m,r)*common_functions.omega(kPrime,m,r)*common_functions.omega(k+kPrime,m,r)*(common_functions.omega(k,m,r) + common_functions.omega(kPrime,m,r) + common_functions.omega(k+kPrime,m,r)))
    
    for k in range(1,l_uv+1):
        sumTerm2 += -pow(g,2)/pow((2*math.pi*r),2)*0.25*(np.heaviside(2*common_functions.omega(k,m,r)+common_functions.omega(2*k,m,r)-e_max,0))/(pow(common_functions.omega(k,m,r),2)*common_functions.omega(2*k,m,r)*(2*common_functions.omega(k,m,r)+common_functions.omega(2*k,m,r)))
        sumTerm3 += -pow(g,2)/pow((2*math.pi*r),2)*0.25*(np.heaviside(m+2*common_functions.omega(k,m,r)-e_max,0))/(pow(common_functions.omega(k,m,r),2)*m*(2*common_functions.omega(k,m,r)+m))

    return sumTerm1+sumTerm2+sumTerm3

def beta1(e_max, m, r, g, l_uv):
    sumTerm1 = 0
    if (e_max > 0 and (e_max - m != 0)):
        for k in range(1,l_uv+1):
            sumTerm1 += (pow(r,3)/e_max)*(np.heaviside(e_max*r-3*k-r*common_functions.omega(k,m,r),0))/( common_functions.omega(k,m,r)* ( pow(r,2)* pow((e_max-common_functions.omega(k,m,r)),2) - pow(k,2) ) )   
        sumTerm1 += 0.5*(r)/(e_max*(pow(e_max-m,2))*m)
        
    sumTerm1 *= -pow(g,2)/(pow(2*math.pi*r,2))
    return sumTerm1
def beta2(e_max, m, r, g, l_uv):
    sumTerm1 = -0.5*beta1(e_max,m,r,g,l_uv)
    sumTerm2 = 0
    sumTerm3 = 0
    for i in range(1,math.floor((l_uv-1)/2)+1):
        for j in range(i+1,l_uv-i+1):
            sumTerm2 += -pow(g,2)/(pow(2*math.pi*r,2))*0.25*np.heaviside(common_functions.omega(i,m,r)+common_functions.omega(j,m,r)+common_functions.omega(i+j,m,r)-e_max,0)/(common_functions.omega(i,m,r)*common_functions.omega(j,m,r)*common_functions.omega(i+j,m,r)*pow(-common_functions.omega(i,m,r)-common_functions.omega(j,m,r)-common_functions.omega(i+j,m,r),2))
    
    for k in range(1,l_uv+1):
        sumTerm3 += -pow(g,2)/(pow(2*math.pi*r,2))*(1/8)*np.heaviside(2*common_functions.omega(k,m,r)+m-e_max,0)/(pow(common_functions.omega(k,m,r),2)*m*pow(-2*common_functions.omega(k,m,r)-m,2))
    
    return sumTerm1 + sumTerm2 + sumTerm3
def lambda2(e_max, m, r, g, l_uv):
    sumTerm1 = 0
    for i in range(1,l_uv+1):
        sumTerm1 += -3*pow(g,2)/(8*math.pi*r)*np.heaviside(2*common_functions.omega(i,m,r)-e_max,0)/(pow(common_functions.omega(i,m,r),3))
    
    return sumTerm1
def alpha1(e_max, m, r, g, l_uv):
    sumTerm = 0
    if (e_max > 0 and (pow(e_max,2) - 4*pow(m,2) > 0)):
        for k in range(1,l_uv+1):
            sumTerm += 3*pow(g,2)/(4*math.pi*r)*0.5*np.heaviside(2*common_functions.omega(k,m,r)-e_max,0)/(pow(-2*common_functions.omega(k,m,r),2)*pow(common_functions.omega(k,m,r),2))
        sumTerm += -3*pow(g,2)/(4*math.pi*r)*r/(pow(e_max,2)*math.sqrt(pow(e_max,2)-4*pow(m,2)))
    else:
        sumTerm = 0
    return sumTerm
def alpha2(e_max,m,g):
    if (e_max > 0 and (pow(e_max,2) - 4*pow(m,2) > 0)):
        sumTerm = -3*pow(g,2)/(2*math.pi)*1/(pow(e_max,2)*math.sqrt(pow(e_max,2)-4*pow(m,2)))
    else:
        sumTerm = 0
    return sumTerm
