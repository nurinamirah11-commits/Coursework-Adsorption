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
from scipy import curve_fit


####################################### Static measurements code ##################################################
# Part 1: Determining the appropriate isotherm model and estimating the isotherm parameters for Romulus adsorbent

# Oxygen and Nitrogen static measurements experimental data
pA_exp=np.array([0.0000, 8106.000, 20265.000, 247435.60, 371153.475, 494871.300, 618690.450])                                #Pa
pB_exp=np.array([0.000, 2026.500, 3039.750, 4053.000, 6079.500, 8106.000, 61909.575, 8106.000, 61909.575,
             111356.175, 173265.750, 247435.650, 321706.875, 395978.100, 470148.000, 544419.225, 618690.450])                #Pa
nA_exp=np.array([0.000, 0.015, 0.035, 0.366, 0.521, 0.693, 0.815])                                                           #mol/kg
nB_exp=np.array([0.000, 0.025, 0.038, 0.053, 0.079, 0.108, 0.562, 0.835, 1.082, 1.295, 1.482, 1.597, 1.714, 1.795, 1.861])   #mol/kg


# Step 1: Identify the single-component isotherm as well as their parameters
def Langmuir_single_component_isotherm(p, n_s, K):
    n_model1_A=(n_s[0]*K[0]*p[0])/(1+K[0]*p[0])
    n_model1_B=(n_s[1]*K[1]*p[1])/(1+K[1]*p[1])
    return [n_model1_A, n_model1_B]

def Freudlich_single_component_isotherm(p, n_s, K, m):
    n_model2_A=(n_s[0]*K[0]*(p[0]**(1/m[0])))/(1+K[0]*(p[0]**(1/m[0])))
    n_model2_B=(n_s[1]*K[1]*(p[1]**(1/m[1])))/(1+K[1]*(p[1]**(1/m[1])))
    return [n_model2_A, n_model2_B]

def Langmuir_Freudlich_single_component_isotherm(p, n_s, K, m):
    n_model3_A=(n_s[0]*K[0]*(p[0]**(1/m[0])))/(1+K[0]*(p[0]**(1/m[0])))
    n_model3_B=(n_s[1]*K[1]*(p[1]**(1/m[1])))/(1+K[1]*(p[1]**(1/m[1])))
    return [n_model3_A, n_model3_B]



# Curve fitting model1:
parameters_guess=[1, 1]  # Initial guess for n_s and K
popt_model1, pcov_model1 = curve_fit(Langmuir_single_component_isotherm, (pA_exp, pB_exp), (nA_exp, nB_exp), p0=parameters_guess)
n_s_model1 = popt_model1[0:2]
K_model1 = popt_model1[2:4]

# Function to determine R^2
def calculate_r_squared(y_exp, y_model):
    residuals = y_exp - y_model
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_exp - np.mean(y_exp))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

# Step 3: Define the isotherm models for a binary component
def Langmuir_single_isotherm(n_s,K,p):
    n_s_A=n_s[0]
    n_s_B=n_s[1]
    K_A=K[0]
    K_B=K[1]
    p_A=p[0]
    p_B=p[1]

    n_A = (n_s_A*K_A*p_A)/(1+K_A*p_A+K_B*p_B)
    n_B = (n_s_B*K_B*p_B)/(1+K_A*p_A+K_B*p_B)

    model1_A=n_A
    model1_B=n_B

    return (model1_A, model1_B)

def Langmuir_dual_isotherm(n_s,K,p):
    n_1_s_A=n_s[0]
    n_2_s_A=n_s[1]
    n_1_s_B=n_s[2]
    n_2_s_B=n_s[3]
    K_1_A=K[0]
    K_2_A=K[1]
    K_1_B=K[2]
    K_2_B=K[3]
    p_A=p[0]
    p_B=p[1]

    n_A = (n_1_s_A*K_1_A*p_A)/(1+K_1_A*p_A+K_1_B*p_B) + (n_2_s_A*K_2_A*p_A)/(1+K_2_A*p_A+K_2_B*p_B)
    n_B=(n_1_s_B*K_1_B*p_B)/(1+K_1_A*p_A+K_1_B*p_B) + (n_2_s_B*K_2_B*p_B)/(1+K_2_A*p_A+K_2_B*p_B)

    model2_A=n_A
    model2_B=n_B

    return (model2_A, model2_B)

def Langmuir_Freudlich_isotherm(n_s,K,p,m):
    n_s_A=n_s[0]
    n_s_B=n_s[1]
    K_A=K[0]
    K_B=K[1]
    p_A=p[0]
    p_B=p[1]

    n_A = (n_s_A*K_A*(p_A**(1/m)))/(1+K_A*(p_A**(1/m))+K_B*(p_B**(1/m)))
    n_B = (n_s_B*K_B*(p_B**(1/m)))/(1+K_A*(p_A**(1/m))+K_B*(p_B**(1/m)))

    model3_A=n_A
    model3_B=n_B

    return (model3_A, model3_B)
