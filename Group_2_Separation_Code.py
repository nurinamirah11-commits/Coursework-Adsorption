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

print("=== REMUS ADSORBENT : STATIC MEASUREMENTS ===")

# Experimental Data for Remus Adsorbent at 303K
# Oxygen (A) and Nitrogen (B)

pA_exp_Re = np.array([0.0000, 33200, 61300, 96100, 143200, 214800, 355900, 548200, 747900, 859000, 931700]) # Pa
pB_exp_Re = np.array([0.0000, 63400, 88700, 117500, 162500, 208500, 303800, 427000, 609500, 776600, 919000]) # Pa
nA_exp_Re = np.array([0.000, 0.0372, 0.0679, 0.1022, 0.1504, 0.2138, 0.3455, 0.5089, 0.6830, 0.7679, 0.8348]) # mol/kg
nB_exp_Re = np.array([0.000, 0.1674, 0.2300, 0.2994, 0.3934, 0.4848, 0.6450, 0.8280, 1.0588, 1.2249, 1.3472]) # mol/kg

# Convert pressure from Pa to bar for fitting purposes
pA_exp_Re = pA_exp_Re / 1e5 # Convert to bar
pB_exp_Re = pB_exp_Re / 1e5 # Convert to bar

# Define Single-Component Isotherm Models
# Model 1 : Langmuir Single Isotherm Model
def langmuir_single_isotherm(p, ns, K):
    return (ns * K * p) / (1.0 + K * p)

# Model 2 : Freundlich Single Isotherm Model
def freundlich_single_isotherm(p, K, m):
    return K * (p ** (1.0/m))

# Model 3 : Langmuir-Freundlich Single Isotherm Model    
def langmuir_freundlich_single_isotherm(p, ns, K, m):
    # Prevents zero-pressure math errors when curve_fit tests extreme parameters (because data starts from p = 0)
    p_safe = np.maximum(p, 1e-9) 
    return (ns * K * (p_safe ** (1.0/m))) / (1.0 + K * (p_safe ** (1.0/m)))

# Fit the Experimental Data to the Isotherm Models for Remus Adsorbent
# Initial guess for parameters : q_sat is the value a bit higher than experimental data, K usually small value (1e-1) in bar-1, m usually greater than 1
# Langmuir model fit

popt_A_Langmuir, pcov_A_Langmuir = curve_fit(langmuir_single_isotherm, pA_exp_Re, nA_exp_Re, p0=[2.0, 1]) # p0=(ns, K)
popt_B_Langmuir, pcov_B_Langmuir = curve_fit(langmuir_single_isotherm, pB_exp_Re, nB_exp_Re, p0=[3.0, 1]) # p0=(ns, K)

# Freundlich model fit

popt_A_Freundlich, pcov_A_Freundlich = curve_fit(freundlich_single_isotherm, pA_exp_Re, nA_exp_Re, p0=[1.0, 1.0]) # p0=(K, m)
popt_B_Freundlich, pcov_B_Freundlich = curve_fit(freundlich_single_isotherm, pB_exp_Re, nB_exp_Re, p0=[1.0, 1.0]) # p0=(K, m)

# Langmuir-Freundlich model fit

popt_A_Langmuir_Freundlich, pcov_A_Langmuir_Freundlich = curve_fit(langmuir_freundlich_single_isotherm, pA_exp_Re, nA_exp_Re, p0=[2.0, 1.0, 1.0], maxfev=10000)
popt_B_Langmuir_Freundlich, pcov_B_Langmuir_Freundlich = curve_fit(langmuir_freundlich_single_isotherm, pB_exp_Re, nB_exp_Re, p0=[3.0, 1.0, 1.0], maxfev=10000) 

# Metrics used to determine the fitting accuracy between all models:
def fitting_accuracy(n_exp, n_model):
    n_average = np.average(n_exp)
    residuals = n_exp - n_model
    residuals_sq = np.sum(residuals ** 2)
    sum_sq = np.sum((n_exp - n_average) ** 2)
    r_squared = 1.0 - residuals_sq / sum_sq
    RMSD = np.sqrt(residuals_sq/n_exp.size)
    return (r_squared, RMSD)

# Calculate the adsorption quantities predicted by each model
n_model_A_langmuir =  langmuir_single_isotherm(pA_exp_Re, *popt_A_Langmuir)
n_model_B_langmuir = langmuir_single_isotherm(pB_exp_Re, *popt_B_Langmuir)

n_model_A_freundlich = freundlich_single_isotherm(pA_exp_Re, *popt_A_Freundlich)
n_model_B_freundlich = freundlich_single_isotherm(pB_exp_Re, *popt_B_Freundlich)

n_model_A_langmuir_freundlich = langmuir_freundlich_single_isotherm(pA_exp_Re, *popt_A_Langmuir_Freundlich)
n_model_B_langmuir_freundlich = langmuir_freundlich_single_isotherm(pB_exp_Re, *popt_B_Langmuir_Freundlich)

# Calculate the fitting accuracy for each model fit
# fitting accuracy for O2 (A)
r_squared_A_langmuir, RMSD_A_langmuir = fitting_accuracy(nA_exp_Re, n_model_A_langmuir)
r_squared_A_freundlich, RMSD_A_freundlich = fitting_accuracy(nA_exp_Re, n_model_A_freundlich)
r_squared_A_langmuir_freundlich, RMSD_A_langmuir_freundlich = fitting_accuracy(nA_exp_Re, n_model_A_langmuir_freundlich)

# fitting accuracy for N2 (B)
r_squared_B_langmuir, RMSD_B_langmuir = fitting_accuracy(nB_exp_Re, n_model_B_langmuir)
r_squared_B_freundlich, RMSD_B_freundlich = fitting_accuracy(nB_exp_Re, n_model_B_freundlich)
r_squared_B_langmuir_freundlich, RMSD_B_langmuir_freundlich = fitting_accuracy(nB_exp_Re, n_model_B_langmuir_freundlich)

# Print the fitted parameters and R-squared values (+RMSD)for each model
print("\n--- Langmuir Model Parameters ---")
print(f"Oxygen (A) -> ns: {popt_A_Langmuir[0]:.4f} mol/kg, K: {popt_A_Langmuir[1]:.4e} bar-1")
print(f"Nitrogen (B) -> ns: {popt_B_Langmuir[0]:.4f} mol/kg, K: {popt_B_Langmuir[1]:.4e} bar-1")

print("\n--- Freundlich Model Parameters ---")
print(f"Oxygen (A) -> K: {popt_A_Freundlich[0]:.4e} bar-1, m: {popt_A_Freundlich[1]:.4f}")
print(f"Nitrogen (B) -> K: {popt_B_Freundlich[0]:.4e} bar-1, m: {popt_B_Freundlich[1]:.4f}")

print("\n--- Langmuir-Freundlich Model Parameters ---")
print(f"Oxygen (A) -> ns: {popt_A_Langmuir_Freundlich[0]:.4f} mol/kg, K: {popt_A_Langmuir_Freundlich[1]:.4e} bar-1, m: {popt_A_Langmuir_Freundlich[2]:.4f}")
print(f"Nitrogen (B) -> ns: {popt_B_Langmuir_Freundlich[0]:.4f} mol/kg, K: {popt_B_Langmuir_Freundlich[1]:.4e} bar-1, m: {popt_B_Langmuir_Freundlich[2]:.4f}")

# R-Squared Comparison (Goodness of Fit)
print("\n--- Model Comparison: R2 ---")
print(f"Langmuir Model      : {r_squared_A_langmuir:.4f} (O2), {r_squared_B_langmuir:.4f} (N2)")
print(f"Freundlich Model    : {r_squared_A_freundlich:.4f} (O2), {r_squared_B_freundlich:.4f} (N2)")
print(f"Langmuir-Freundlich : {r_squared_A_langmuir_freundlich:.4f} (O2), {r_squared_B_langmuir_freundlich:.4f} (N2)")

# RMSD Comparison (Error Magnitude)
print("\n--- Model Comparison: RMSD (mol/kg) ---")
print(f"Langmuir Model RMSD            : {RMSD_A_langmuir:.4f} mol/kg (O2), {RMSD_B_langmuir:.4f} mol/kg (N2)")
print(f"Freundlich Model RMSD          : {RMSD_A_freundlich:.4f} mol/kg (O2), {RMSD_B_freundlich:.4f} mol/kg (N2)")
print(f"Langmuir-Freundlich (Sips) RMSD: {RMSD_A_langmuir_freundlich:.4f} mol/kg (O2), {RMSD_B_langmuir_freundlich:.4f} mol/kg (N2)")

# Parity plot - Experimental vs Estimated data
# Langmuir model parity plot
plt.plot(nA_exp_Re, n_model_A_langmuir, marker='o', ls='none', ms=8, color='k', mfc="#87F090", label='Langmuir (Oxygen)')
plt.plot(nB_exp_Re, n_model_B_langmuir, marker='s', ls='none', ms=8, color='k', mfc="#48F132", label='Langmuir (Nitrogen)')

# Freundlich model parity plot
plt.plot(nA_exp_Re, n_model_A_freundlich, marker='o', ls='none', ms=8, color='k', mfc="#87C6F0", label='Freundlich (Oxygen)')
plt.plot(nB_exp_Re, n_model_B_freundlich, marker='s', ls='none', ms=8, color='k', mfc="#279FF0", label='Freundlich (Nitrogen)') 

# Langmuir-Freundlich model parity plot
plt.plot(nA_exp_Re, n_model_A_langmuir_freundlich, marker='o', ls='none', ms=8, color='k', mfc="#EC7F7F", label='Langmuir-Freundlich (Oxygen)')
plt.plot(nB_exp_Re, n_model_B_langmuir_freundlich, marker='s', ls='none', ms=8, color='k', mfc="#EC1919", label='Langmuir-Freundlich (Nitrogen)')

# Show plot
plt.plot([0, 2.1], [0, 2.1], 'k--', lw=0.5)  # Plot diagonal
plt.xlim([0, 2.1])
plt.ylim([0, 2.1])
plt.xlabel(r'$n \quad {\rm (Experimental)}$', size=16)
plt.ylabel(r'$n \quad {\rm (Predicted)}$', size=16)
plt.legend(fontsize=10)
plt.savefig('fitting_rate.pdf', bbox_inches='tight')
plt.grid(color='0.8', linestyle='-', linewidth=1)
plt.show()