# Test from https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
# See how to adapt to odes : https://medium.com/@octaviogl69/parameter-estimation-for-differential-equations-part-i-ordinary-differential-equations-443c6ba112ae
# Numpy array that contains the integration times

# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.optimize import curve_fit
#
# def func(x, a, b, c):
#     return a * np.exp(-b * x) + c
#
# y.t = np.linspace(0, 4, 50)
# y = func(y.t, 2.5, 1.3, 0.5)
# np.random.seed(1729)
# y_noise = 0.2 * np.random.normal(size=y.t.size)
# ydata = y + y_noise
# plt.plot(y.t, ydata, 'b-', label='data')
#
# # Fit for the parameters a, b, c of the function func:
# popt, pcov = curve_fit(func, y.t, ydata)
#
# plt.plot(y.t, func(y.t, *popt), 'r-',
#         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
#
# # Constrain the optimization to the region of 0 <= a <= 3, 0 <= b <= 1 and 0 <= c <= 0.5:
# popt, pcov = curve_fit(func, y.t, ydata, bounds=(0, [3., 1., 0.5]))
# plt.plot(y.t, func(y.t, *popt), 'g--',
#          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
#
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()

# ------------------------------------------------------------------------------------------
#
# import numpy as np
# from scipy.integrate import solve_ivp
# import matplotlib.pyplot as plt
#
# # Numpy array that contains the integration times
# kc = 2.
#
# def model(t,C):
#     return -kc * C[0]
#
# # Solution of the model
# x0=[2.]
# z = solve_ivp(model,[0,2],x0,t_eval= np.linspace(0,2,15))
#
# # Plot the solution
# plt.plot(z.t, z.y[0], 'r-x', label=r'Initial solution')
# plt.show()
# ------------------------------------------------------------------------------------------

import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

kc= 2.
C0= 2.
nSamples= 30

# General function to solve the ODE model
def GeneralSolver(t, k, C0):

    localK = k
    localC0 = C0

    def ODEModel(C, t):
        return -localK * C

    sol = odeint(ODEModel, localC0, t)

    return sol[:, 0]

# General function to solve the ODE model
def GeneralSolver_ivp(t, k, C0):

    localK = k
    localC0 = C0

    def ODEModel(C, t):
        return -localK * C

    #sol = odeint(ODEModel, localC0, t)
    tSpan = [t[0], t[-1]]

    #               fun,     t_span,    y0,     t_eval
    sol = solve_ivp(ODEModel, tSpan, [localC0], t_eval=np.linspace(t[0], t[-1], nSamples))
    r = np.transpose(sol.y)[:,0]
    return r


# Solves the ODE model using the initial condition provided above
def ODESolution(t, k):
    return GeneralSolver_ivp(t, k, C0)


# Element wise sum of two iterables of the same size, name makes reference to the output rather than the process
def MakeNoisyData(Data, Noise):
    return [val + cal for val, cal in zip(Data, Noise)]


# Solving the ODE model
t = np.linspace(0, 2, num=nSamples)
y = ODESolution(t, kc)

# # Making some simulated data to perform regression
WhiteNoise = [np.random.uniform(low=-1, high=1) / 10 for val in y]
y_noise = MakeNoisyData(y, WhiteNoise)


# Parameter estimation : uses curve_fit that finds the
# best parameter to fit a given fcn to data. The trick is
# to encapsulate the Ode solver into a fnc
Kp = curve_fit(ODESolution, t, y_noise)[0][0]
print("Kp= ", Kp)
fitSolution = ODESolution(t, Kp)

# Plot the solution ( no curve fitting )
plt.plot(t, y, 'r-', label='Initial solution')
plt.plot(t, y_noise, 'g-', label='Noisy solution')
plt.plot(t, fitSolution, 'b-', label='Fit Solution')

plt.show()
