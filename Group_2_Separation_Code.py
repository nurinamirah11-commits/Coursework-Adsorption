# Separation Coursework 2026/2027
# Group 2 members:
# Muhammad Daniel Bin Surali (Student ID: 11429231)
# Nurin Amirah Binti Azmi (Student ID: XXXXXXXX)
#
# Main objectives: 
# Selecting the optimal adsorbent for oxygen concentrators based on the static measurements and dynamic measurements

# Library required for the codes
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


####################################### Static measurements code ##################################################
# Part 1: Determining the appropriate isotherm model and estimating the isotherm parameters for both adsorbents (ROMULUS and REMUS).

# ROMULUS adsorbent experimental isotherm data for Oxygen (A) and Nitrogen (B) at 303K
pA_exp_Ro=np.array([0.0000, 8106.000, 20265.000, 247435.650, 371153.475, 494871.300, 618690.450])                               #Pa
pB_exp_Ro=np.array([0.000, 2026.500, 3039.750, 4053.000, 6079.500, 8106.000, 61909.575, 111356.175,
                     173265.750, 247435.650, 321706.875, 395978.100, 470148.000, 544419.225, 618690.450])                       #Pa
nA_exp_Ro=np.array([0.000, 0.015, 0.035, 0.366, 0.521, 0.693, 0.815])                                                           #mol/kg
nB_exp_Ro=np.array([0.000, 0.025, 0.038, 0.053, 0.079, 0.108, 0.562, 0.835, 1.082, 1.295, 1.482, 1.597, 1.714, 1.795, 1.861])   #mol/kg

# Step 1: Identify the single-component isotherm as well as their parameters for Oxygen and Nitrogen for ROMULUS adsorbent
def Langmuir_single_component_isotherm(p,ns, K):                        # Langmuir single-component isotherm function
    return (ns*K*p)/(1+K*p)

def Freudlich_single_component_isotherm(p, K, m):                   # Freudlich single-component isotherm function
    return (K*(p**(1/m)))

def Langmuir_Freudlich_single_component_isotherm(p, ns, K, m):          # Langmuir-Freudlich single-component isotherm function
    return (ns*K*(p**(1/m)))/(1+K*(p**(1/m)))

def model1_A_Ro_fit(p,nsA_Ro_model1,KA_Ro_model1):
    
    nA_Ro_model1=Langmuir_single_component_isotherm(pA_exp_Ro, nsA_Ro_model1, KA_Ro_model1)

    return nA_Ro_model1

def model1_B_Ro_fit(p, nsB_Ro_model1, KB_Ro_model1):

    nB_Ro_model1=Langmuir_single_component_isotherm(pB_exp_Ro,nsB_Ro_model1,KB_Ro_model1)

    return(nB_Ro_model1)

def model2_A_Ro_fit(p, KA_Ro_model2,mA_Ro_model2):

    nA_Ro_model2=Freudlich_single_component_isotherm(pA_exp_Ro,KA_Ro_model2,mA_Ro_model2)

    return(nA_Ro_model2)

def model2_B_Ro_fit(p, KB_Ro_model2, mB_Ro_model2):

    nB_Ro_model2=Freudlich_single_component_isotherm(pB_exp_Ro, KB_Ro_model2, mB_Ro_model2)

    return (nB_Ro_model2)

def model3_A_Ro_fit(p, nsA_Ro_model3, KA_Ro_model3, mA_Ro_model3):

    nA_Ro_model3=Langmuir_Freudlich_single_component_isotherm(pA_exp_Ro,nsA_Ro_model3, KA_Ro_model3, mA_Ro_model3)

    return (nA_Ro_model3)

def model3_B_Ro_fit(p, nsB_Ro_model3,KB_Ro_model3, mB_Ro_model3):

    nB_Ro_model3=Langmuir_Freudlich_single_component_isotherm(pB_exp_Ro,nsB_Ro_model3,KB_Ro_model3, mB_Ro_model3)

    return (nB_Ro_model3)

# Curve fitting MODEL 1 for Oxygen (ROMULUS)
parameters_guess=[1,1]
popt_model1_A, pcov_model1_A=curve_fit(model1_A_Ro_fit,pA_exp_Ro,nA_exp_Ro,p0=parameters_guess)
nsA_Ro_model1, KA_Ro_model1 = popt_model1_A
y1A=Langmuir_single_component_isotherm(pA_exp_Ro,nsA_Ro_model1,KA_Ro_model1) 

print(y1A)
print(nsA_Ro_model1,KA_Ro_model1)

# Curve fitting MODEL 1 for Nitrogen (ROMULUS)
parameters_guess=[1,1]
popt_model1_B, pcov_model1_B=curve_fit(model1_B_Ro_fit,pB_exp_Ro,nB_exp_Ro,p0=parameters_guess)
nsB_Ro_model1, KB_Ro_model1 = popt_model1_B
y1B=Langmuir_single_component_isotherm(pB_exp_Ro,nsB_Ro_model1,KB_Ro_model1)

print(y1B)
print(nsB_Ro_model1,KB_Ro_model1)

# Curve fitting MODEL 2 for Oxygen (ROMULUS)
parameters_guess=[1,1]
popt_model2_A, pcov_model2_A=curve_fit(model2_A_Ro_fit,pA_exp_Ro,nA_exp_Ro,p0=parameters_guess)
nsA_Ro_model2, mA_Ro_model2 = popt_model2_A
y2A=Freudlich_single_component_isotherm(pA_exp_Ro,nsA_Ro_model2,mA_Ro_model2) 

print(y2A)
print(nsA_Ro_model2,mA_Ro_model2)

# Curve fitting MODEL 2 for Nitrogen (ROMULUS)
parameters_guess=[1,1]
popt_model2_B, pcov_model2_B=curve_fit(model2_B_Ro_fit,pB_exp_Ro,nB_exp_Ro,p0=parameters_guess)
nsB_Ro_model2, mB_Ro_model2 = popt_model2_B
y2B=Freudlich_single_component_isotherm(pB_exp_Ro,nsB_Ro_model2,mB_Ro_model2) 

print(y2B)
print(nsB_Ro_model2,mB_Ro_model2)

# Curve fitting MODEL 3 for Oxygen (ROMULUS)
parameters_guess=[1, 1, 1]
popt_model3_A, pcov_model3_A=curve_fit(model3_A_Ro_fit,pA_exp_Ro,nA_exp_Ro,p0=parameters_guess)
nsA_Ro_model3, kA_Ro_model3 ,mA_Ro_model3 = popt_model3_A
y3A=Langmuir_Freudlich_single_component_isotherm(pB_exp_Ro, nsA_Ro_model3, kA_Ro_model3, mA_Ro_model3) 

print(y3A)
print(nsA_Ro_model3, kA_Ro_model3 ,mA_Ro_model3)

# Curve fitting MODEL 3 for Nitrogen (ROMULUS)
parameters_guess=[1, 1, 1]
popt_model3_B, pcov_model3_B=curve_fit(model3_B_Ro_fit,pB_exp_Ro,nB_exp_Ro,p0=parameters_guess)
nsB_Ro_model3, kB_Ro_model3 ,mB_Ro_model3 = popt_model3_B
y3B=Langmuir_Freudlich_single_component_isotherm(pB_exp_Ro, nsB_Ro_model3, kB_Ro_model3, mB_Ro_model3) 

print(y3B)
print(nsB_Ro_model3, kB_Ro_model3 ,mB_Ro_model3)

