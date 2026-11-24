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
    p_safe = np.maximum(p, 1e-9)   # avoids problems at p = 0 because without this, the output gives WARNING
    return K * (p_safe ** (1.0/m))

# Model 3 : Langmuir-Freundlich Single Isotherm Model    
def langmuir_freundlich_single_isotherm(p, ns, K, m):
    p_safe = np.maximum(p, 1e-9)   # avoids problems at p = 0 because without this, the output gives WARNING
    return (ns * K * (p_safe ** (1.0/m))) / (1.0 + K * (p_safe ** (1.0/m)))

# Model 4 : Dual-Site Langmuir Isotherm Model (DSL)
def dual_site_langmuir_isotherm(p, ns1, K1, ns2, K2):
    site1 = (ns1 * K1 * p) / (1.0 + K1 * p)
    site2 = (ns2 * K2 * p) / (1.0 + K2 * p)
    return site1 + site2

# Fit the Experimental Data to the Isotherm Models for Remus Adsorbent
# Initial guess for parameters : q_sat is the value a bit higher than experimental data, K usually small value (1e-1) in bar-1, m usually greater than 1
# Adding bounds avoid the absurd values for LF (to be safe)

# Langmuir model fit
popt_A_Langmuir, pcov_A_Langmuir = curve_fit(langmuir_single_isotherm, pA_exp_Re, nA_exp_Re,
    p0=[2.0, 0.1], bounds=([1e-6, 1e-8], [50.0, np.inf]), maxfev=20000)
popt_B_Langmuir, pcov_B_Langmuir = curve_fit(langmuir_single_isotherm, pB_exp_Re, nB_exp_Re,
    p0=[3.0, 0.1], bounds=([1e-6, 1e-8], [50.0, np.inf]), maxfev=20000)

# Freundlich model fit
popt_A_Freundlich, pcov_A_Freundlich = curve_fit(freundlich_single_isotherm, pA_exp_Re, nA_exp_Re,
    p0=[0.1, 1.1], bounds=([1e-8, 0.1], [50.0, 20.0]), maxfev=20000)
popt_B_Freundlich, pcov_B_Freundlich = curve_fit(freundlich_single_isotherm, pB_exp_Re, nB_exp_Re,
    p0=[0.2, 1.2], bounds=([1e-8, 0.1], [50.0, 20.0]), maxfev=20000)

# Langmuir-Freundlich model fit
popt_A_Langmuir_Freundlich, pcov_A_Langmuir_Freundlich = curve_fit(langmuir_freundlich_single_isotherm, pA_exp_Re, nA_exp_Re,
    p0=[2.0, 0.1, 1.0], bounds=([1e-6, 1e-8, 0.1], [20.0, np.inf, 20.0]), maxfev=30000)
popt_B_Langmuir_Freundlich, pcov_B_Langmuir_Freundlich = curve_fit(langmuir_freundlich_single_isotherm, pB_exp_Re, nB_exp_Re,
    p0=[3.0, 0.1, 1.2], bounds=([1e-6, 1e-8, 0.1], [20.0, np.inf, 20.0]), maxfev=30000)

# Dual-Site Langmuir model fit
popt_A_DSL, pcov_A_DSL = curve_fit(dual_site_langmuir_isotherm, pA_exp_Re, nA_exp_Re,
    p0=[1.0, 0.2, 1.5, 0.05], bounds=([1e-6, 1e-8, 1e-6, 1e-8], [20.0, np.inf, 20.0, np.inf]), maxfev=50000)
popt_B_DSL, pcov_B_DSL = curve_fit(dual_site_langmuir_isotherm, pB_exp_Re, nB_exp_Re,
    p0=[1.5, 0.3, 2.0, 0.08], bounds=([1e-6, 1e-8, 1e-6, 1e-8], [20.0, np.inf, 20.0, np.inf]), maxfev=50000)

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
n_model_A_langmuir = langmuir_single_isotherm(pA_exp_Re, *popt_A_Langmuir)
n_model_B_langmuir = langmuir_single_isotherm(pB_exp_Re, *popt_B_Langmuir)

n_model_A_freundlich = freundlich_single_isotherm(pA_exp_Re, *popt_A_Freundlich)
n_model_B_freundlich = freundlich_single_isotherm(pB_exp_Re, *popt_B_Freundlich)

n_model_A_langmuir_freundlich = langmuir_freundlich_single_isotherm(pA_exp_Re, *popt_A_Langmuir_Freundlich)
n_model_B_langmuir_freundlich = langmuir_freundlich_single_isotherm(pB_exp_Re, *popt_B_Langmuir_Freundlich)

n_model_A_DSL = dual_site_langmuir_isotherm(pA_exp_Re, *popt_A_DSL)
n_model_B_DSL = dual_site_langmuir_isotherm(pB_exp_Re, *popt_B_DSL)

# Calculate the fitting accuracy for each model fit
r_squared_A_langmuir, RMSD_A_langmuir = fitting_accuracy(nA_exp_Re, n_model_A_langmuir)
r_squared_A_freundlich, RMSD_A_freundlich = fitting_accuracy(nA_exp_Re, n_model_A_freundlich)
r_squared_A_langmuir_freundlich, RMSD_A_langmuir_freundlich = fitting_accuracy(nA_exp_Re, n_model_A_langmuir_freundlich)
r_squared_A_DSL, RMSD_A_DSL = fitting_accuracy(nA_exp_Re, n_model_A_DSL)

r_squared_B_langmuir, RMSD_B_langmuir = fitting_accuracy(nB_exp_Re, n_model_B_langmuir)
r_squared_B_freundlich, RMSD_B_freundlich = fitting_accuracy(nB_exp_Re, n_model_B_freundlich)
r_squared_B_langmuir_freundlich, RMSD_B_langmuir_freundlich = fitting_accuracy(nB_exp_Re, n_model_B_langmuir_freundlich)
r_squared_B_DSL, RMSD_B_DSL = fitting_accuracy(nB_exp_Re, n_model_B_DSL)

# Print the fitted parameters and R-squared values (+RMSD) for each model
print("====== REMUS ADSORBENT : STATIC MEASUREMENTS ======")

print("\nLangmuir Model Parameters")
print(f"O2 (A) -> ns: {popt_A_Langmuir[0]:.4f} mol/kg, K: {popt_A_Langmuir[1]:.4e} bar-1")
print(f"N2 (B) -> ns: {popt_B_Langmuir[0]:.4f} mol/kg, K: {popt_B_Langmuir[1]:.4e} bar-1")

print("\nFreundlich Model Parameters")
print(f"O2 (A) -> K: {popt_A_Freundlich[0]:.4e} bar-1, m: {popt_A_Freundlich[1]:.4f}")
print(f"N2 (B) -> K: {popt_B_Freundlich[0]:.4e} bar-1, m: {popt_B_Freundlich[1]:.4f}")

print("\nLangmuir-Freundlich Model Parameters")
print(f"O2 (A) -> ns: {popt_A_Langmuir_Freundlich[0]:.4f} mol/kg, K: {popt_A_Langmuir_Freundlich[1]:.4e} bar-1, m: {popt_A_Langmuir_Freundlich[2]:.4f}")
print(f"N2 (B) -> ns: {popt_B_Langmuir_Freundlich[0]:.4f} mol/kg, K: {popt_B_Langmuir_Freundlich[1]:.4e} bar-1, m: {popt_B_Langmuir_Freundlich[2]:.4f}")

print("\nDual-Site Langmuir Model Parameters")
print(f"O2 (A) -> ns1: {popt_A_DSL[0]:.4f} mol/kg, K1: {popt_A_DSL[1]:.4e} bar-1, "
      f"ns2: {popt_A_DSL[2]:.4f} mol/kg, K2: {popt_A_DSL[3]:.4e} bar-1")
print(f"N2 (B) -> ns1: {popt_B_DSL[0]:.4f} mol/kg, K1: {popt_B_DSL[1]:.4e} bar-1, "
      f"ns2: {popt_B_DSL[2]:.4f} mol/kg, K2: {popt_B_DSL[3]:.4e} bar-1")

# R-Squared Comparison (Goodness of Fit)
print("\nModel Comparison: R2")
print(f"Langmuir Model            : {r_squared_A_langmuir:.5f} (O2), {r_squared_B_langmuir:.5f} (N2)")
print(f"Freundlich Model          : {r_squared_A_freundlich:.5f} (O2), {r_squared_B_freundlich:.5f} (N2)")
print(f"Langmuir-Freundlich Model : {r_squared_A_langmuir_freundlich:.5f} (O2), {r_squared_B_langmuir_freundlich:.5f} (N2)")
print(f"Dual-Site Langmuir Model  : {r_squared_A_DSL:.5f} (O2), {r_squared_B_DSL:.5f} (N2)")

# RMSD Comparison (Error Magnitude)
print("\nModel Comparison: RMSD (mol/kg)")
print(f"Langmuir Model            : {RMSD_A_langmuir:.4f} mol/kg (O2), {RMSD_B_langmuir:.4f} mol/kg (N2)")
print(f"Freundlich Model          : {RMSD_A_freundlich:.4f} mol/kg (O2), {RMSD_B_freundlich:.4f} mol/kg (N2)")
print(f"Langmuir-Freundlich Model : {RMSD_A_langmuir_freundlich:.4f} mol/kg (O2), {RMSD_B_langmuir_freundlich:.4f} mol/kg (N2)")
print(f"Dual-Site Langmuir Model  : {RMSD_A_DSL:.4f} mol/kg (O2), {RMSD_B_DSL:.4f} mol/kg (N2)")

# Best model for Remus: Langmuir (2 parameters, R2 > 0.999 for both gases, saturates, extends to competitive form)
# Standard errors = square root of the diagonal of the covariance matrix : to evaluate how well ns and K are determined by the data
error_A_Re = np.sqrt(np.diag(pcov_A_Langmuir))   # [error in ns, error in K] for O2
error_B_Re = np.sqrt(np.diag(pcov_B_Langmuir))   # [error in ns, error in K] for N2

# Store directly the parameters determined in variables to be used in Part 2
ns_A_Re, K_A_Re = popt_A_Langmuir
ns_B_Re, K_B_Re = popt_B_Langmuir

print("\n====== BEST MODEL FOR REMUS: LANGMUIR ======")
print("Parameters value +/- standard error:")
print(f"Oxygen (A)   -> ns = {ns_A_Re:.3f} +/- {error_A_Re[0]:.3f} mol/kg ({100*error_A_Re[0]/ns_A_Re:.1f}%), "
      f"K = {K_A_Re:.4f} +/- {error_A_Re[1]:.4f} bar-1 ({100*error_A_Re[1]/K_A_Re:.1f}%)")
print(f"Nitrogen (B) -> ns = {ns_B_Re:.3f} +/- {error_B_Re[0]:.3f} mol/kg ({100*error_B_Re[0]/ns_B_Re:.1f}%), "
      f"K = {K_B_Re:.4f} +/- {error_B_Re[1]:.4f} bar-1 ({100*error_B_Re[1]/K_B_Re:.1f}%)")

# Evaluate the static performance of an adsorbent using its Langmuir parameters (can be used for Romulus too)
def evaluate_adsorbent(adsorbent, ns_A, K_A, ns_B, K_B, P_process = 1.0):
    # P_process = operating pressure in bar (100 kPa = 1 bar)

    # Henry constants (initial slope, for low pressure) and selectivity (also at low pressure)
    H_A = ns_A * K_A   # mol/kg/bar for O2
    H_B = ns_B * K_B   # mol/kg/bar for N2
    alpha_low = H_B / H_A

    # Capacity at the operating pressure (1 bar)
    nA_process = langmuir_single_isotherm(P_process, ns_A, K_A)   # mol/kg of O2
    nB_process = langmuir_single_isotherm(P_process, ns_B, K_B)   # mol/kg of N2

    # Selectivity N2/O2 at the operating pressure
    alpha = nB_process / nA_process

    print(f"\n====== STATIC PERFORMANCE: {adsorbent} ======")
    print(f"Henry constant O2 : {H_A:.4f} mol/kg/bar")
    print(f"Henry constant N2 : {H_B:.4f} mol/kg/bar")
    print(f"Selectivity N2/O2 at low pressure : {alpha_low:.2f}")
    print(f"\nO2 capacity at {P_process} bar : {nA_process:.4f} mol/kg")
    print(f"N2 capacity at {P_process} bar : {nB_process:.4f} mol/kg")
    print(f"Selectivity N2/O2 at {P_process} bar   : {alpha:.2f}")

    return {"H_A": H_A, "H_B": H_B, "Selecivity (low P)": alpha_low,
            "nA_op": nA_process, "nB_op": nB_process, "Selectivity (process P)": alpha}

# Call it for Remus
results_Re = evaluate_adsorbent("Remus", popt_A_Langmuir[0], popt_A_Langmuir[1],
                                popt_B_Langmuir[0], popt_B_Langmuir[1])

# Plotting results: experimental points and Langmuir model lines in one graph
p_smooth_A = np.linspace(0, max(pA_exp_Re), 200)   # smooth pressure range for O2
p_smooth_B = np.linspace(0, max(pB_exp_Re), 200)   # smooth pressure range for N2

plt.figure(figsize=(7, 5))   # 7 in x 5 in = maximum figure size allowed in the coursework

# Oxygen (A): points = experiment, line = Langmuir
plt.plot(pA_exp_Re, nA_exp_Re, 'bo', label='O2 experimental', markersize=6)
plt.plot(p_smooth_A, langmuir_single_isotherm(p_smooth_A, *popt_A_Langmuir), 'b-', label='O2 Langmuir', lw=2)

# Nitrogen (B): points = experiment, line = Langmuir
plt.plot(pB_exp_Re, nB_exp_Re, 'rs', label='N2 experimental', markersize=6)
plt.plot(p_smooth_B, langmuir_single_isotherm(p_smooth_B, *popt_B_Langmuir), 'r-', label='N2 Langmuir', lw=2)

plt.xlabel('Pressure [bar]', fontsize=11)
plt.ylabel('Adsorption quantity [mol/kg]', fontsize=11)
plt.title('Remus: O2 and N2 adsorption isotherms at 303 K', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('remus_isotherms.pdf', bbox_inches='tight')
plt.show()