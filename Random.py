# Random line:


# REMUS adsorbent experimental isotherm data for Oxygen (A) and Nitrogen (B) at 303K
pA_exp_Re=np.array([0.0000, 33200, 61300, 96100, 143200, 214800, 355900, 548200, 747900, 859000, 931700])                       #Pa
pB_exp_Re=np.array([0.000, 63400, 88700, 117500, 162500, 208500, 303800, 427000, 609500, 776600, 919000])                       #Pa
nA_exp_Re=np.array([0.000, 0.0372, 0.0679, 0.1022, 0.1504, 0.2138, 0.3455, 0.5089, 0.6830, 0.7679, 0.8348])                     #mol/kg
nB_exp_Re=np.array([0.000, 0.1674, 0.2300, 0.2994, 0.3934, 0.4848, 0.6450, 0.8280, 1.0588, 1.2249, 1.3472 ])                    #mol/kg

#Combining experimental data into two independent set
p_combined=(pA_exp_Ro, pB_exp_Ro, pA_exp_Re, pB_exp_Re)
n_combined=np.concatenate((nA_exp_Ro, nB_exp_Ro, nA_exp_Re, nB_exp_Re))

# Another Random lines:

def model1_fit(p, nsA_Ro_model1, nsB_Ro_model1, nsA_Re_model1, nsB_Re_model1, KA_Ro_model1, KB_Ro_model1, KA_Re_model1, KB_Re_model1):
    
    pA_exp_Ro, pB_exp_Ro, pA_exp_Re, pB_exp_Re=p
    
    nA_Ro_model1 = Langmuir_single_component_isotherm(pA_exp_Ro, nsA_Ro_model1, KA_Ro_model1)
    nB_Ro_model1 = Langmuir_single_component_isotherm(pB_exp_Ro, nsB_Ro_model1, KB_Ro_model1)
    nA_Re_model1 = Langmuir_single_component_isotherm(pA_exp_Re, nsA_Re_model1, KA_Re_model1)
    nB_Re_model1 = Langmuir_single_component_isotherm(pB_exp_Re, nsB_Re_model1, KB_Re_model1)

    return np.concatenate((nA_Ro_model1, nB_Ro_model1, nA_Re_model1, nB_Re_model1))


def model2_fit(p, nsA_Ro_model2, nsB_Ro_model2, nsA_Re_model2, nsB_Re_model2, KA_Ro_model2, KB_Ro_model2, KA_Re_model2, KB_Re_model2,mA_Ro_model2, mb_Ro_model2, mA_Re_model2, mB_Re_model2):
    
    pA_exp_Ro, pB_exp_Ro, pA_exp_Re, pB_exp_Re=p
    
    nA_Ro_model2 = Freudlich_single_component_isotherm(pA_exp_Ro, nsA_Ro_model2, KA_Ro_model2, mA_Ro_model2)
    nB_Ro_model2 = Freudlich_single_component_isotherm(pB_exp_Ro, nsB_Ro_model2, KB_Ro_model2, mb_Ro_model2)
    nA_Re_model2 = Freudlich_single_component_isotherm(pA_exp_Re, nsA_Re_model2, KA_Re_model2, mA_Re_model2)
    nB_Re_model2 = Freudlich_single_component_isotherm(pB_exp_Re, nsB_Re_model2, KB_Re_model2, mB_Re_model2)

    return np.concatenate((nA_Ro_model2, nB_Ro_model2, nA_Re_model2, nB_Re_model2))

def model3_fit(p, nsA_Ro_model3, nsB_Ro_model3, nsA_Re_model3, nsB_Re_model3, KA_Ro_model3, KB_Ro_model3, KA_Re_model3, KB_Re_model3,mA_Ro_model3, mb_Ro_model3, mA_Re_model3, mB_Re_model3):
    
    pA_exp_Ro, pB_exp_Ro, pA_exp_Re, pB_exp_Re=p
    
    nA_Ro_model3 = Langmuir_Freudlich_single_component_isotherm(pA_exp_Ro, nsA_Ro_model3, KA_Ro_model3, mA_Ro_model3)
    nB_Ro_model3 = Langmuir_Freudlich_single_component_isotherm(pB_exp_Ro, nsB_Ro_model3, KB_Ro_model3, mb_Ro_model3)
    nA_Re_model3 = Langmuir_Freudlich_single_component_isotherm(pA_exp_Re, nsA_Re_model3, KA_Re_model3, mA_Re_model3)
    nB_Re_model3 = Langmuir_Freudlich_single_component_isotherm(pB_exp_Re, nsB_Re_model3, KB_Re_model3, mB_Re_model3)

    return np.concatenate((nA_Ro_model3, nB_Ro_model3, nA_Re_model3, nB_Re_model3))

# Curve fitting model1_fit:
parameters_guess=[1, 1, 1, 1, 1, 1, 1, 1]  # Initial guess for KA_Ro, KB_Ro, KA_Re, KB_Re
popt_model1, pcov_model1 = curve_fit(model1_fit, p_combined, n_combined, p0=parameters_guess)

# Curve fitting model2_fit:
parameters_guess=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # Initial guess for KA_Ro, KB_Ro, KA_Re, KB_Re, mA_Ro, mb_Ro, mA_Re, mB_Re
popt_model2, pcov_model2 = curve_fit(model2_fit, p_combined, n_combined, p0=parameters_guess)

# Curve fitting model3_fit:
parameters_guess=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # Initial guess for KA_Ro, KB_Ro, KA_Re, KB_Re, mA_Ro, mb_Ro, mA_Re, mB_Re
popt_model3, pcov_model3 = curve_fit(model3_fit, p_combined, n_combined, p0=parameters_guess)



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
