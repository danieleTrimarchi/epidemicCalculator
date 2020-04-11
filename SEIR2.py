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
#   Im: Portion of the infected showing mild symptoms
#   Ih: Portion of the infected requiring hospitalisation
#   Id: Portion of the infected that will die
#   R : Recoveded individuals
#   Rm: Individuals recoveded from mild symptoms
#   Rh: Individuals recoveded from hospitalisation
#   Rd: Death Individuals
idx = ['S','E','I','Im','Ih','Id','R','Rm','Rh','Rd']

# -- INITIAL CONDITIONS -- :
N = 7.e6    # Population
Tinc = 5.2  # Incubation period [days]
Tinf = 2.9  # Time the patient is infectious [days]
Rt = 2.2    # Reproduction number (how many people infected by each person)
pd = 0.02   # Case death rate. Percentage of the infected patients who will die
ph = 0.2    # Hospitalization rate. Percentage of the infected patients who will be hospitalized
Tmild = 11.1 # Recovery time for mild cases
THosp = 28.6 # Length of hospital stay [days]
TDeath= 32   # Time from end of incubation to death [days]

beta = Rt / Tinf   #
gamma = 1. / Tinf  # infectious rate [1/days]
sigma = 1. / Tinc  # incubation rate [1/days]
gamma_m = 1. / Tmild
gamma_h = 1. / THosp
gamma_d = 1. / TDeath
pm = 1. - ph - pd   # Mild rate. Percentage of the infected patients with mild symptoms


# -- EQUATIONS --
#   dS(t)/dt = - beta S(t) I(t)
#   dE(t)/dt = beta S(t) I(t) - sigma E(t)
# --
#   dI(t)/dt = sigma E(t) - gamma I(t)
#  this is the sum of three separated contributions:
#   dIm(t)/dt = pm * gamma I(t) - gamma_m * Im(t)

# \!/ ----------------------------------------------------- \!/
# \!/ ADD Severe_H \!/
#   dIh(t)/dt = ph * gamma I(t) - gamma_h * Ih(t)
# this is severe. to be added: the severe hospitalized.
# There is a lag between the severe and its hospitalisation
# \!/ ----------------------------------------------------- \!/

#   dId(t)/dt = pd * gamma I(t) - gamma_d * Id(t)
# --
#   dR(t)/dt = gamma I(t)
#  this is the sum of three separated contributions:
#   dRm(t)/dt = gamma_m Im(t)
#   dRh(t)/dt = gamma_h Ih(t)
#   dRd(t)/dt = gamma_d Id(t)


# Remember: z=[S,E,I,Im,Ih,Id,R,Rm,Rh,Rd]
def model(t,z):

    S = z[idx.index('S')]
    E = z[idx.index('E')]
    I = z[idx.index('I')]
    Im = z[idx.index('Im')]
    Ih = z[idx.index('Ih')]
    Id = z[idx.index('Id')]
    R = z[idx.index('R')]
    Rm = z[idx.index('Rm')]
    Rh = z[idx.index('Rh')]
    Rd = z[idx.index('Rd')]

    #print("R0= ",(a*beta)/((mu+a)*mu+gamma))

    dSdt = - beta * I * S / N
    dEdt = beta * I * S / N - sigma * E

    dIdt = sigma * E - gamma * I
    dImdt = pm * gamma * I - gamma_m * Im
    dIhdt = ph * gamma * I - gamma_h * Ih
    dIddt = pd * gamma * I - gamma_d * Id

    dRdt = gamma * I
    dRmdt = gamma_m * Im
    dRhdt = gamma_h * Ih
    dRddt = gamma_d * Id

    dzdt= [dSdt,dEdt,dIdt,dImdt,dIhdt,dIddt,dRdt,dRmdt,dRhdt,dRddt]
    return dzdt


S0 = N      # Initial population
E0 = 0.     # Exposed individuals @t0
I0 = 1.     # Infected individuals @t0
R0 = 0.

#    [S0, E0, I0,   Im0,  Ih0,  Id0, R0,  Rm0,  Rh0,  Rd0]
y0 = [S0, E0, I0, I0*pm,I0*ph,I0*pd, R0,R0*pm,I0*ph,I0*pd]

# solve ODE in the timespan t=[0,200]
z = solve_ivp(model,[0,200],y0)

# plot results
plt.subplot(211)

plt.plot(z.t,z.y[idx.index('S')],'b-',label=r'S')
plt.plot(z.t,z.y[idx.index('E')],'y-',label=r'E')
plt.plot(z.t,z.y[idx.index('I')],'r-',label=r'I')
plt.plot(z.t,z.y[idx.index('R')],'g-',label=r'R')
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
plt.plot(z.t,z.y[idx.index('Rd')],'k-',label=r'Rd')
plt.plot(ref[0],ref[5],'kx',label=r'$fatalities_{ref}$')
plt.legend(loc='center left',
           fontsize='xx-small')

plt.ylabel('Fatalities')
plt.xlabel('time [days]')

plt.show()
