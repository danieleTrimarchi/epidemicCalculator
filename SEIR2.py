# SEIR model without vital dynamics
# https://www.idmod.org/docs/hiv/model-seir.html
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# function that returns the ode to solve:

# -- VARIABLES --
#   S : Susceptibles individuals, or individuals that can be infected (@t0 this is the whole population)
#   E : Exposed individuals (infected but not yet infectious)
#   I : Infected individuals
#   R : Recoveded individuals

# -- INITIAL CONDITIONS -- :
N = 7.e6    # Population
Tinc = 5.2  # Incubation period
Tinf = 2.9  # Time the patient is infectious
Rt = 2.2    # Reproduction number (how many people infected by each person)

beta = Rt / Tinf   #
gamma = 1. / Tinf  # infectious rate [1/days]
sigma = 1. / Tinc  # incubation rate [1/days]

# -- EQUATIONS --
#   dS(t)/dt = - beta S(t) I(t)
#   dE(t)/dt = beta S(t) I(t) - sigma E(t)
#   dI(t)/dt = sigma E(t) - gamma I(t)
#   dR(t)/dt = gamma I(t)

# Remember: z=[S,E,I,R]
def model(t,z):

    S = z[0]
    E = z[1]
    I = z[2]
    R = z[3]

    #print("R0= ",(a*beta)/((mu+a)*mu+gamma))

    dSdt = - beta * I * S / N
    dEdt = beta * I * S / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I

    dzdt= [dSdt,dEdt,dIdt,dRdt]
    return dzdt


S0 = N      # Initial population
E0 = 0.     # Exposed individuals @t0
I0 = 1.     # Infected individuals @t0
R0 = 0.
y0 = [S0, E0, I0, R0]

# solve ODE in the timespan t=[0,200]
z = solve_ivp(model,[0,200],y0)

# plot results
plt.subplot(211)

plt.plot(z.t,z.y[0],'b-',label=r'S')
plt.plot(z.t,z.y[1],'y-',label=r'E')
plt.plot(z.t,z.y[2],'r-',label=r'I')
plt.plot(z.t,z.y[3],'g-',label=r'R')
plt.ylabel('response')


#plt.yscale("log")

ref = np.loadtxt('gabgo_data',
                    skiprows=1,
                    unpack=True)
plt.plot(ref[0],ref[1],'bx',label=r'$S_{ref}$')
plt.plot(ref[0],ref[2],'yx',label=r'$E_{ref}$')
plt.plot(ref[0],ref[3],'rx',label=r'$I_{ref}$')
plt.plot(ref[0],ref[4],'gx',label=r'$R_{ref}$')

plt.title(r'R_0=$\frac{\beta} {\gamma}=$'+ str(beta/(gamma)))
plt.legend(loc='center left',
           fontsize='xx-small')

plt.subplot(212)

# Compute the fatalities
# f = int((infectious@t+shift)*fatRate)
#        fatRate=0.02
#        shift= 32 days
fatRate = 0.02
shift = 32
fatalities = [0] * len(z.y[0])
for i in range(len(z.y[0])):
    y = z.y[2].tolist()[0:i]
    fatalities[i] = np.trapz(y) * fatRate

plt.plot(z.t,fatalities,'k-',label=r'Fatalities')
plt.plot(ref[0],ref[5],'kx',label=r'$fatalities_{ref}$')
plt.legend(loc='center left',
           fontsize='xx-small')

plt.ylabel('Fatalities')
plt.xlabel('time [days]')

plt.show()
