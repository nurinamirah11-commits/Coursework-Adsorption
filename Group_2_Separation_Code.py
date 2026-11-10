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


################### Static measurements code ###############################
# Part 1: Determining the appropriate isotherm model and estimating the isotherm parameters
# Step 1: Define the isotherm models

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

# Step 2: Inserting the experienmental data for the static measurements
# i) Experimental data for the static measurements for Romulus adsorbent

P_Romulus_A = np.array([[0.000, 8106.000, 20265.000, 247435.650, 371153.475, 494871.300, 618690.450]])
P_Romulus_B = np.array([[0.000, 8106.000, 20265.000, 247435.650, 371153.475, 494871.300, 618690.450]])
