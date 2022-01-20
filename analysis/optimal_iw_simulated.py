from classes.piv import PIV
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rc('text', usetex=True)
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
plt.rcParams['text.usetex']=True

iws = range(1, 40)
stderrs = []
v_x_mean = []
v_y_mean = []

for iw in iws:
    v_x = v_y = []
    for filename in os.listdir("../data/simulated/iw_investigation_gradient"):
        piv = PIV(f"../data/simulated/iw_investigation_gradient/{filename}", iw, 16, 80, 0, "5pointgaussian", False)
        v_x.append(piv.correlation_averaged_velocity_field[0][0, 0, 2])
        v_y.append(piv.correlation_averaged_velocity_field[0][0, 0, 3])
    v_x_mean.append(np.mean(v_x))
    stderrs.append(np.std(v_x))
    """plt.title(f"Interrogation window: {iw}; Standard Error: {np.std(v_x):.2f}")
    plt.hist2d(v_x, v_y, bins = 50)
    plt.show()"""



def powerfit(x, m, c, n):
    return m * x**(n) + c


xs = np.linspace(np.min(iws), np.max(iws), 100)
bias = np.abs(np.array(v_x_mean))

fit = curve_fit(powerfit, iws, bias, sigma = stderrs / np.sqrt(len(stderrs)), maxfev = 10000)
print(fit[0])
print(np.sqrt(np.diag(fit[1])))

plt.title("Relationship between interrogation window size and bias")
plt.xlabel("Interrogation window size (px)")
plt.ylabel("Bias on the mean (px)")
plt.scatter(iws, bias)
plt.plot(xs, powerfit(xs, *fit[0]), c = "black", ls = "--")
plt.show()


def powerfit(x, m, c, n):
    return m * x**(-n) + c


fit = curve_fit(powerfit, iws, stderrs)
print(fit[0])
print(np.sqrt(np.diag(fit[1])))

plt.title("Relationship between interrogation window size and standard error")
plt.xlabel("Interrogation window size (px)")
plt.ylabel("Standard Error (px)")
plt.scatter(iws, stderrs)
plt.plot(xs, powerfit(xs, *fit[0]), c = "black", ls = "--")
plt.show()