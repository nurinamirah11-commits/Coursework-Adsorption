# Separation Coursework 2026/2027
# Group 2 members:
# Muhammad Daniel Bin Surali (Student ID: 11429231)
# Nurin Amirah Binti Azmi (Student ID: 14539377)
#
# Main objectives: 
# Selecting the optimal adsorbent for oxygen concentrators based on the static measurements and dynamic measurements

# Library required for the codes
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

print("===REMUS ADSORBENT : STATIC MEASUREMENTS===")

# Step 1 : Experimental Data for Remus Adsorbent at 303K
# Oxygen (A) and Nitrogen (B)

pA_exp_Re = np.array([0.0000, 33200, 61300, 96100, 143200, 214800, 355900, 548200, 747900, 859000, 931700]) # Pa
pB_exp_Re = np.array([0.0000, 63400, 88700, 117500, 162500, 208500, 303800, 427000, 609500, 776600, 919000]) # Pa
nA_exp_Re = np.array([0.000, 0.0372, 0.0679, 0.1022, 0.1504, 0.2138, 0.3455, 0.5089, 0.6830, 0.7679, 0.8348]) # mol/kg
nB_exp_Re = np.array([0.000, 0.1674, 0.2300, 0.2994, 0.3934, 0.4848, 0.6450, 0.8280, 1.0588, 1.2249, 1.3472]) # mol/kg

# Step 2 : Define Single-Component Isotherm Models
# Model 1 : Langmuir Single Isotherm Model
def langmuir_single_isotherm(p, ns, K):
    return (ns * K * p) / (1.0 + K * p)

# Model 2 : Freundlich Single Isotherm Model
def freundlich_single_isotherm(p, K, m):
    return K * (p ** (1.0/m))

# Model 3 : Langmuir-Freundlich Single Isotherm Model    
def langmuir_freundlich_single_isotherm(p, ns, K, m):
    return (ns * K * (p ** (1.0/m))) / (1.0 + K * (p ** (1.0/m)))

# Step 3 : Fit the Experimental Data to the Isotherm Models for Remus Adsorbent
# Initial guess for parameters : q_sat is the value a bit higher than experimental data, K usually small value (1e-6) in Pa-1, m usually greater than 1
# Langmuir model fit

popt_A_Langmuir, pcov_A_Langmuir = curve_fit(langmuir_single_isotherm, pA_exp_Re, nA_exp_Re, p0=[2.0, 1e-6]) # p0=(ns, K)
popt_B_Langmuir, pcov_B_Langmuir = curve_fit(langmuir_single_isotherm, pB_exp_Re, nB_exp_Re, p0=[3.0, 1e-6]) # p0=(ns, K)

# Freundlich model fit

popt_A_Freundlich, pcov_A_Freundlich = curve_fit(freundlich_single_isotherm, pA_exp_Re, nA_exp_Re, p0=[1e-6, 1.0]) # p0=(K, m)
popt_B_Freundlich, pcov_B_Freundlich = curve_fit(freundlich_single_isotherm, pB_exp_Re, nB_exp_Re, p0=[1e-6, 1.0]) # p0=(K, m)

# Langmuir-Freundlich model fit

popt_A_Langmuir_Freundlich, pcov_A_Langmuir_Freundlich = curve_fit(langmuir_freundlich_single_isotherm, pA_exp_Re, nA_exp_Re, p0=[2.0, 1e-6, 1.0], bounds=([0, 0, 0], [5.0, 1.0, 5.0]), maxfev=10000)
popt_B_Langmuir_Freundlich, pcov_B_Langmuir_Freundlich = curve_fit(langmuir_freundlich_single_isotherm, pB_exp_Re, nB_exp_Re, p0=[3.0, 1e-6, 1.0], bounds=([0, 0, 0], [5.0, 1.0, 5.0]), maxfev=10000) 

# Step 4 : Evaluate R-squared for each model fit
def calculate_r_squared(y_exp, y_model):
    residuals = y_exp - y_model
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_exp - np.mean(y_exp)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot)
    return r_squared

# Calculate the adsorption quantities predicted by each model
n_model_A_langmuir =  langmuir_single_isotherm(pA_exp_Re, *popt_A_Langmuir)
n_model_B_langmuir = langmuir_single_isotherm(pB_exp_Re, *popt_B_Langmuir)

n_model_A_freundlich = freundlich_single_isotherm(pA_exp_Re, *popt_A_Freundlich)
n_model_B_freundlich = freundlich_single_isotherm(pB_exp_Re, *popt_B_Freundlich)

n_model_A_langmuir_freundlich = langmuir_freundlich_single_isotherm(pA_exp_Re, *popt_A_Langmuir_Freundlich)
n_model_B_langmuir_freundlich = langmuir_freundlich_single_isotherm(pB_exp_Re, *popt_B_Langmuir_Freundlich)

# Calculate R-squared for each model fit
# R squared for O2 (A)
r_squared_A_langmuir = calculate_r_squared(nA_exp_Re, n_model_A_langmuir)
r_squared_A_freundlich = calculate_r_squared(nA_exp_Re, n_model_A_freundlich)
r_squared_A_langmuir_freundlich = calculate_r_squared(nA_exp_Re, n_model_A_langmuir_freundlich)

# R squared for N2 (B)
r_squared_B_langmuir = calculate_r_squared(nB_exp_Re, n_model_B_langmuir)
r_squared_B_freundlich = calculate_r_squared(nB_exp_Re, n_model_B_freundlich)
r_squared_B_langmuir_freundlich = calculate_r_squared(nB_exp_Re, n_model_B_langmuir_freundlich)

# Step 5 : Print the fitted parameters and R-squared values for each model
print("\n--- Langmuir Model Parameters ---")
print(f"Oxygen (A) -> q_sat: {popt_A_Langmuir[0]:.4f}, K: {popt_A_Langmuir[1]:.4e}")
print(f"Nitrogen (B) -> q_sat: {popt_B_Langmuir[0]:.4f}, K: {popt_B_Langmuir[1]:.4e}")

print("\n--- Freundlich Model Parameters ---")
print(f"Oxygen (A) -> K: {popt_A_Freundlich[0]:.4e}, m: {popt_A_Freundlich[1]:.4f}")
print(f"Nitrogen (B) -> K: {popt_B_Freundlich[0]:.4e}, m: {popt_B_Freundlich[1]:.4f}")

print("\n--- Langmuir-Freundlich Model Parameters ---")
print(f"Oxygen (A) -> q_sat: {popt_A_Langmuir_Freundlich[0]:.4f}, K: {popt_A_Langmuir_Freundlich[1]:.4e}, m: {popt_A_Langmuir_Freundlich[2]:.4f}")
print(f"Nitrogen (B) -> q_sat: {popt_B_Langmuir_Freundlich[0]:.4f}, K: {popt_B_Langmuir_Freundlich[1]:.4e}, m: {popt_B_Langmuir_Freundlich[2]:.4f}")

print("\n--- Model R-squared Comparison ---")
print(f"\n======= Oxygen (A) R-squared Values ===")
print(f"Langmuir R2:                 {r_squared_A_langmuir:.4f}")
print(f"Freundlich R2:               {r_squared_A_freundlich:.4f}")
print(f"Langmuir-Freundlich R2:      {r_squared_A_langmuir_freundlich:.4f}")

print(f"\n======= Nitrogen (B) R-squared Values ===")
print(f"Langmuir R2:                 {r_squared_B_langmuir:.4f}")
print(f"Freundlich R2:               {r_squared_B_freundlich:.4f}")
print(f"Langmuir-Freundlich R2:      {r_squared_B_langmuir_freundlich:.4f}")

