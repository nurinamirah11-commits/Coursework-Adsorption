
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ==================================================
# 1. EXPERIMENTAL DATA
# ==================================================

xA = np.array([1, 2, 3])
xB = np.array([4, 5])
xC = np.array([6, 7, 8, 9])

yA = np.array([2, 4, 6])
yB = np.array([8, 10])
yC = np.array([12, 14, 16, 18])


# ==================================================
# 2. COMBINE THE DATASETS
# ==================================================

# Combine all x-values into one array
x_combined = np.concatenate((xA, xB, xC))

# Combine all y-values into one array
y_combined = np.concatenate((yA, yB, yC))

# Create dataset identifiers
dataset = np.concatenate((
    np.zeros(len(xA)),
    np.ones(len(xB)),
    np.full(len(xC), 2)
))

# Display combined data
print("\n--- COMBINED EXPERIMENTAL DATA ---")
print("Combined x-values:", x_combined)
print("Combined y-values:", y_combined)
print("Dataset identifiers:", dataset)


# ==================================================
# 3. COMBINE X-VALUES AND DATASET IDENTIFIERS
# ==================================================

# Create a 2D array
xdata = np.vstack((x_combined, dataset))

print("\n--- XDATA ARRAY ---")
print(xdata)

print("\nFirst row (x-values):", xdata[0])
print("Second row (dataset identifiers):", xdata[1])


# ==================================================
# 4. DEFINE THE COMBINED MODEL
# ==================================================

def model(xdata, mA, cA, mB, cB, mC, cC):

    # Extract x-values and dataset identifiers
    x = xdata[0]
    group = xdata[1]

    # Create an empty array for predicted y-values
    y = np.empty(len(x))

    # Dataset A: y = mA*x + cA
    maskA = group == 0
    y[maskA] = mA*x[maskA] + cA

    # Dataset B: y = mB*x + cB
    maskB = group == 1
    y[maskB] = mB*x[maskB] + cB

    # Dataset C: y = mC*x + cC
    maskC = group == 2
    y[maskC] = mC*x[maskC] + cC

    return y


# ==================================================
# 5. PERFORM CURVE FITTING
# ==================================================

# Initial parameter guesses
p0 = [1, 1, 1, 1, 1, 1]

# Optimise parameters
popt, pcov = curve_fit(
    model,
    xdata,
    y_combined,
    p0=p0
)


# ==================================================
# 6. EXTRACT OPTIMISED PARAMETERS
# ==================================================

mA, cA, mB, cB, mC, cC = popt

print("\n--- OPTIMISED PARAMETERS ---")
print(f"Dataset A: m = {mA:.4f}, c = {cA:.4f}")
print(f"Dataset B: m = {mB:.4f}, c = {cB:.4f}")
print(f"Dataset C: m = {mC:.4f}, c = {cC:.4f}")


# ==================================================
# 7. GENERATE PREDICTED VALUES
# ==================================================

yA_fit = mA*xA + cA
yB_fit = mB*xB + cB
yC_fit = mC*xC + cC

print("\n--- PREDICTED Y-VALUES ---")
print("Dataset A:", yA_fit)
print("Dataset B:", yB_fit)
print("Dataset C:", yC_fit)


# ==================================================
# 8. PLOT EXPERIMENTAL DATA AND FITTED MODELS
# ==================================================

plt.figure(figsize=(9, 6))

# Experimental data
plt.scatter(xA, yA, label="Dataset A experimental")
plt.scatter(xB, yB, label="Dataset B experimental")
plt.scatter(xC, yC, label="Dataset C experimental")

# Fitted models
plt.plot(xA, yA_fit, label="Dataset A fitted")
plt.plot(xB, yB_fit, label="Dataset B fitted")
plt.plot(xC, yC_fit, label="Dataset C fitted")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Simultaneous Curve Fitting of Three Datasets")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
