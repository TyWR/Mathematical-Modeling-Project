from pde_resolve import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec



# Macros
kappa = 1                # Kappa
m = 3                    # m = 3
xmax = 1                 # Window maximum size



subplot = False

if (subplot) :
    myPde = pde(1, m, 0.001, xmax)

    # Init of the q-function
    #myPde.initConstantQ(-1)
    myPde.initFunctionQ(1, -3, 1, 5)

    gs = GridSpec(5, 1)
    plt.subplot(gs[0:3, :])
    # Resolving the stationary problem
    myPde.resolveStationary(0, True)

    # Compute the flow and plotting it on the graph
    myPde.computeFlow(0.05, 150, True)

    plt.subplot(gs[3:5, :])
    myPde.plotQ()

    plt.tight_layout()
    plt.show()

else :
    # myPde = pde(1, m, 0.001, xmax)
    # myPde.initConstantQ(-1)
    # myPde.resolveStationary(1, False)
    # myPde.computePerturbation(0.1, 0.001, 500, 100)
    # plt.legend()
    # plt.show()

    myPde = pde(1, m, 0.001)

    # Init of the q-function
    myPde.initFunctionQ(1, -3, 1, 5)

    # Resolving the stationary problem
    myPde.resolveStationary(0, False)

    myPde.initFunctionQ(1, -3, 0.5, 4.5)
    result = np.array(myPde.computeChange(0.0001, 2000, 3000))

    t, x = np.meshgrid(myPde.t, myPde.x)


    plt.pcolormesh(t, x, result)
    plt.show()
