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


####################################### Static measurements code ##################################################
# Part 1: Determining the appropriate isotherm model and estimating the isotherm parameters for both adsorbents (ROMULUS and REMUS).

# ROMULUS adsorbent experimental isotherm data for Oxygen (A) and Nitrogen (B) at 303K
pA_exp_Ro_raw=np.array([0.0000, 33200, 61300, 96100, 143200, 214800, 355900, 548200, 747900, 859000, 931700])     #Pa
pB_exp_Ro_raw=np.array([0.0000, 63400, 88700, 117500, 162500, 208500, 303800, 427000, 609500, 776600, 919000])       #Pa                  

nA_exp_Ro=np.array([0.000, 0.0372, 0.0679, 0.1022, 0.1504, 0.2138, 0.3455, 0.5089, 0.6830, 0.7679, 0.8348])    #mol/kg
nB_exp_Ro=np.array([0.000, 0.1674, 0.2300, 0.2994, 0.3934, 0.4848, 0.6450, 0.8280, 1.0588, 1.2249, 1.3472])   #mol/kg

pA_exp_Ro=pA_exp_Ro_raw/101325             #bar
pB_exp_Ro=pB_exp_Ro_raw/101325             #bar

# Step 1: Identify the single-component isotherm as well as their parameters for Oxygen and Nitrogen for ROMULUS adsorbent
def Langmuir_single_component_isotherm(p,ns, K):                         # Langmuir single-component isotherm function
    return (ns*K*p)/(1+K*p)

def Freudlich_single_component_isotherm(p, K, m):                        # Freudlich single-component isotherm function
    return (K*(p**(1/m)))

def Langmuir_Freudlich_single_component_isotherm(p, ns, K, m):           # Langmuir-Freudlich single-component isotherm function
    return (ns*K*(p**(1/m)))/(1+K*(p**(1/m)))

# Metrics used to determine the fitting accuracy between all models:
def fitting_accuracy(n_exp, n_est):
    n_average=np.average(n_exp)
    residuals_sq=np.sum((n_exp-n_est)**2)
    sum_sq=np.sum((n_exp-n_average)**2)
    rsquare=1-residuals_sq/sum_sq
    RMSD=np.sqrt(residuals_sq/n_exp.size)
    return (rsquare,RMSD)

# Fitting functions for all models:

def model1_A_Ro_fit(p,nsA_Ro_model1,KA_Ro_model1):
    
    nA_Ro_model1=Langmuir_single_component_isotherm(pA_exp_Ro, nsA_Ro_model1, KA_Ro_model1)

    return (nA_Ro_model1)

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

# Curve fitting MODEL 1 (Oxygen)
parameters_guess=[1, 1]
popt_model1_A, pcov_model1_A=curve_fit(model1_A_Ro_fit,pA_exp_Ro,nA_exp_Ro,p0=parameters_guess)
nsA_Ro_model1, KA_Ro_model1 = popt_model1_A
y1A=Langmuir_single_component_isotherm(pA_exp_Ro,nsA_Ro_model1,KA_Ro_model1) 

# Calculate error metrics for MODEL 1 fitting (Oxygen)
rsquare, RMSD = fitting_accuracy(nA_exp_Ro, y1A)

# Print MODEL 1 fitting results (Oxygen)
print('\n--- Langmuir (Oxygen) ---')
print('ns: {0:12.6f}'.format(nsA_Ro_model1))
print('K: {0:12.6f}'.format(KA_Ro_model1))
print('R2: {0:12.6f}'.format(rsquare))
print('RMSD: {0:12.6f}'.format(RMSD))

# Parity plot for MODEL 1 fitting (Oxygen) - Experimental vs Estimated data
plt.plot(nA_exp_Ro, y1A, marker='o', ls='none', ms=8, color='k', mfc="#87F090", label='Langmuir (Oxygen)')

# Curve fitting MODEL 1 (Nitrogen)
parameters_guess=[1, 1]
popt_model1_B, pcov_model1_B=curve_fit(model1_B_Ro_fit,pB_exp_Ro,nB_exp_Ro,p0=parameters_guess)
nsB_Ro_model1, KB_Ro_model1 = popt_model1_B
y1B=Langmuir_single_component_isotherm(pB_exp_Ro, nsB_Ro_model1, KB_Ro_model1)

# Calculate error metrics for MODEL 1 fitting (Nitrogen)
rsquare, RMSD = fitting_accuracy(nB_exp_Ro, y1B)

# Print MODEL 1 fitting results (Nitrogen)
print('\n--- Langmuir (Nitrogen) ---')
print('ns: {0:12.6f}'.format(nsB_Ro_model1))
print('K: {0:12.6f}'.format(KB_Ro_model1))
print('R2: {0:12.6f}'.format(rsquare))
print('RMSD: {0:12.6f}'.format(RMSD))

# Parity plot for MODEL 1 fitting (Nitrogen) - Experimental vs Estimated data
plt.plot(nB_exp_Ro, y1B, marker='o', ls='none', ms=8, color='k', mfc="#48F132", label='Langmuir (Nitrogen)')

# Curve fitting MODEL 2 (Oxygen)
parameters_guess=[1, 1]
popt_model2_A, pcov_model2_A=curve_fit(model2_A_Ro_fit,pA_exp_Ro,nA_exp_Ro,p0=parameters_guess)
KA_Ro_model2, mA_Ro_model2 = popt_model2_A
y2A=Freudlich_single_component_isotherm(pA_exp_Ro, KA_Ro_model2, mA_Ro_model2) 

# Calculate error metrics for MODEL 2 fitting (Oxygen)
rsquare, RMSD = fitting_accuracy(nA_exp_Ro, y2A)

# Print MODEL 1 fitting results (Oxygen)
print('\n--- Freudlich (Oxygen) ---')
print('K: {0:12.6f}'.format(KA_Ro_model2))
print('m: {0:12.6f}'.format(mA_Ro_model2))
print('R2: {0:12.6f}'.format(rsquare))
print('RMSD: {0:12.6f}'.format(RMSD))

# Parity plot for MODEL2 fitting (Oxygen) - Experimental vs Estimated data
plt.plot(nA_exp_Ro, y2A, marker='o', ls='none', ms=8, color='k', mfc="#87C6F0", label='Freudlich (Oxygen)')

# Curve fitting MODEL 2 (Nitrogen)
parameters_guess=[1, 1]
popt_model2_B, pcov_model2_B=curve_fit(model2_B_Ro_fit,pB_exp_Ro,nB_exp_Ro,p0=parameters_guess)
KB_Ro_model2, mB_Ro_model2 = popt_model2_B
y2B=Freudlich_single_component_isotherm(pB_exp_Ro, KB_Ro_model2, mB_Ro_model2) 

# Calculate error metrics for MODEL 2 fitting (Nitrogen)
rsquare, RMSD = fitting_accuracy(nB_exp_Ro, y2B)

# Print MODEL 2 fitting results (Nitrogen)
print('\n--- Freudlich (Nitrogen) ---')
print('K: {0:12.6f}'.format(KB_Ro_model2))
print('m: {0:12.6f}'.format(mB_Ro_model2))
print('R2: {0:12.6f}'.format(rsquare))
print('RMSD: {0:12.6f}'.format(RMSD))

# Parity plot for MODEL 2 fitting (Nitrogen) - Experimental vs Estimated data
plt.plot(nB_exp_Ro, y2B, marker='o', ls='none', ms=8, color='k', mfc="#279FF0", label='Freudlich (Nitrogen)')

# Curve fitting MODEL 3 (Oxygen)
parameters_guess=[1, 1, 1]
popt_model3_A, pcov_model3_A=curve_fit(model3_A_Ro_fit,pA_exp_Ro,nA_exp_Ro,p0=parameters_guess, maxfev = 5000)
nsA_Ro_model3, KA_Ro_model3 ,mA_Ro_model3 = popt_model3_A
y3A=Langmuir_Freudlich_single_component_isotherm(pA_exp_Ro, nsA_Ro_model3, KA_Ro_model3, mA_Ro_model3) 

# Calculate error metrics for MODEL 3 fitting (Oxygen)
rsquare, RMSD = fitting_accuracy(nA_exp_Ro, y3A)

# Print MODEL 3 fitting results (Oxygen)
print('\n--- Langmuir-Freudlich (Oxygen) ---')
print('ns: {0:12.6f}'.format(nsA_Ro_model3))
print('K: {0:12.6f}'.format(KA_Ro_model3))
print('m:{0:12.6f}'.format(mA_Ro_model3))
print('R2: {0:12.6f}'.format(rsquare))
print('RMSD: {0:12.6f}'.format(RMSD))

# Parity plot for MODEL 3 fitting (Oxygen) - Experimental vs Estimated data
plt.plot(nA_exp_Ro, y3A, marker='o', ls='none', ms=8, color='k', mfc="#EC7F7F", label='Langmuir-Freudlich (Oxygen)')

# Curve fitting MODEL 3 (Nitrogen)
parameters_guess=[1, 1, 1]
popt_model3_B, pcov_model3_B=curve_fit(model3_B_Ro_fit,pB_exp_Ro,nB_exp_Ro,p0=parameters_guess, maxfev=5000)
nsB_Ro_model3, KB_Ro_model3 ,mB_Ro_model3 = popt_model3_B
y3B=Langmuir_Freudlich_single_component_isotherm(pB_exp_Ro, nsB_Ro_model3, KB_Ro_model3, mB_Ro_model3) 

# Calculate error metrics for MODEL 3 fitting (Nitrogen)
rsquare, RMSD = fitting_accuracy(nB_exp_Ro, y3B)

# Print MODEL 3 fitting results (Nitrogen)
print('\n--- Langmuir-Freudlich (Nitrogen) ---')
print('ns: {0:12.6f}'.format(nsB_Ro_model3))
print('K: {0:12.6f}'.format(KB_Ro_model3))
print('m:{0:12.6f}'.format(mB_Ro_model3))
print('R2: {0:12.6f}'.format(rsquare))
print('RMSD: {0:12.6f}'.format(RMSD))

# Parity plot for MODEL 3 fitting (Nitrogen) - Experimental vs Estimated data
plt.plot(nB_exp_Ro, y3B, marker='o', ls='none', ms=8, color='k', mfc="#EC1919", label='Langmuir-Freudlich (Nitrogen)')

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