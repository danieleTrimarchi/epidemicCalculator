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
idx = ['S','E','I','Im','Is','Id','Ish','R','Rm','Rs','Rd']
def i(dofStr):
    return idx.index(dofStr)

# -- INITIAL CONDITIONS -- :
N = 7.e6    # Population
Tinc = 5.2  # Incubation period [days]
Tinf = 2.9  # Time the patient is infectious [days]
Rt = 2.2    # Reproduction number (how many people infected by each person)
pd = 0.02   # Case death rate. Percentage of the infected patients who will die
ps = 0.2    # Hospitalization rate. Percentage of the infected patients who will be hospitalized
pm = 1. - ps - pd   # Mild rate. Percentage of the infected patients with mild symptoms
Tmild = 11.1 # Recovery time for mild cases
TSevere = 28.6 # Length of hospital stay [days]
THospLag = 5 # Lag between Infection of severe case and hospitalisation [days]
TDeath= 32   # Time from end of incubation to death [days]

# -- EQUATIONS --
#   dS(t)/dt = -(Rt/Tinf) * S(t) I(t)
#   dE(t)/dt =  (Rt/Tinf) * S(t) I(t) - 1/Tinc E(t)
# --
#   dI(t)/dt = 1/Tinc * E(t) - 1/Tinf I(t)
#   this is the sum of three separated contributions:
#       dIm(t)/dt = pm * 1/Tinf * I(t) - 1/Tmild     * Im(t)
#       dIs(t)/dt = ps * 1/Tinf * I(t) - 1/T_hospLag * Is(t)
#       dId(t)/dt = pd * 1/Tinf * I(t) - 1/Tdead     * Id(t)
# --
#  Hp: mild cases recover
#     some of the severe cases will be hospitalized after some time lag:
#       dIsh(t)/dt = 1/T_hospLag * Is(t) - 1/Tsevere * Ish(t)
#     deadly cases just recover when dying
# --
#  dR(t)/dt = 1/Tinf I(t)
#  this is the sum of three separated contributions:
#   dRm(t)/dt = 1/Tmild   * Im(t)
#   dRs(t)/dt = 1/Tsevere * Ish(t)
#   dRd(t)/dt = 1/Tdead   * Id(t)

# def x = [       S              E          I        Im         Is         Id        Ish        R    Rm   Rs   Rd ]^T
# and A =
#  S      [-Rt/(Tinf*N)*I(t)                                                                                      ]
#  E      [ Rt/(Tinf*N)*I(t)  -1/Tinc                                                                             ]
#  I      [                    1/Tinc    -1/Tinf                                                                  ]
#  Im     [                             pm*1/Tinf  -1/Tmild                                                       ]
#  Is     [                             ps*1/Tinf           -1/T_hospLag                                          ]
#  Id     [                             pd*1/Tinf                        -1/Tdead                                 ]
#  Ish    [                                                  1/T_hospLag         -1/Tsevere                       ]
#  R      [                               1/Tinf                                                0                 ]
#  Rm     [                                         1/Tmild                                          0            ]
#  Rs     [                                                                       1/Tsevere               0       ]
#  Rd     [                                                               1/Tdead                              0  ]
A = np.zeros((len(idx), len(idx)))
# --
A[i('E'),i('E')] = -1. / Tinc
A[i('I'),i('E')] =  1. / Tinc
# --
A[i('I'),i('I')]  =     -1. / Tinf
A[i('Im'),i('I')] = pm * 1. / Tinf
A[i('Is'),i('I')] = ps * 1. / Tinf
A[i('Id'),i('I')] = pd * 1. / Tinf
A[i('R'),i('I')]  =      1. / Tinf
# --
A[i('Im'),i('Im')] = -1. / Tmild
A[i('Rm'),i('Im')] =  1. / Tmild
# --
A[i('Is'),i('Is')] = -1. / THospLag
A[i('Ish'),i('Is')]=  1. / THospLag
# --
A[i('Id'),i('Id')] = -1. / TDeath
A[i('Rd'),i('Id')] =  1. / TDeath

A[i('Ish'),i('Ish')]= -1. / TSevere
A[i('Rs'),i('Ish')] =  1. / TSevere

def model(t,z):

    # Update non-const part of the matrix A
    A[i('S'),i('S')]= - Rt/(Tinf*N) * z[i('I')]
    A[i('E'),i('S')]=   Rt/(Tinf*N) * z[i('I')]

    return np.dot(A,z)


S0 = N      # Initial population
E0 = 0.     # Exposed individuals @t0
I0 = 1.     # Infected individuals @t0
R0 = 0.

#     S  E  I  Im    Is    Id   Ish R  Rm    Rs    Rd
x0 = [S0,E0,I0,I0*pm,I0*ps,I0*pd,I0,R0,R0*pm,R0*ps,R0*pd]

# solve ODE in the timespan t=[0,200]
z = solve_ivp(model,[0,200],x0)

# plot results
plt.subplot(211)

plt.plot(z.t,z.y[i('S')],'b-',label=r'S')
plt.plot(z.t,z.y[i('E')],'y-',label=r'E')
plt.plot(z.t,z.y[i('I')],'r-',label=r'I')
plt.plot(z.t,z.y[i('R')],'g-',label=r'R')
plt.ylabel('response')

#plt.yscale("log")

ref = np.loadtxt('gabgo_data',
                    skiprows=1,
                    unpack=True)
plt.plot(ref[0],ref[1],'bx',label=r'$S_{ref}$')
plt.plot(ref[0],ref[2],'yx',label=r'$E_{ref}$')
plt.plot(ref[0],ref[3],'rx',label=r'$I_{ref}$')
plt.plot(ref[0],ref[4],'gx',label=r'$R_{ref}$')

plt.title(r'R_0=$'+ str(Rt))
plt.legend(loc='center left',
           fontsize='xx-small')

plt.subplot(212)

# Compute the fatalities
plt.plot(z.t,z.y[i('Rd')],'k-',label=r'Rd')
plt.plot(ref[0],ref[5],'kx',label=r'$fatalities_{ref}$')
plt.legend(loc='center left',
           fontsize='xx-small')

plt.ylabel('Fatalities')
plt.xlabel('time [days]')

plt.show()
