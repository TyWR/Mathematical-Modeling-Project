# Main source file with the resolving of the PDE
import numpy as np
import matplotlib.pyplot as plt
import types

class pde :

    # -------------------------------------------------------------
    # This class represents the equation of the form :
    # dt(h) + lambda * dx(h^m+2) = q
    # -------------------------------------------------------------

    def __init__(self, kappa, m, step) :
        # different parameters of the equation

        self.q = 0                                      # q function
        self.m = m                                      # m parameter
        self.k = kappa                                  # Kappa parameter
        self.l = -(kappa / (m+1)) * (1/(m+2) - 1)       # Lambda parameter
        self.n = 0                                      # Number of points
        self.u = []                                     # Values of height
        self.dx = step                                  # Space step (dx)
        self.x = [i*self.dx for i in range(self.n)]     # Grid of x

        self.vx = []                                    # Velocity along x axis
        self.vy = []                                    # Velocity along y axis
        self.coordx = []                                # Coordinates along x of velocity vector
        self.coordy = []                                # Coordinates along y of velocity vector
        self.p = []                                     # Values of the gradient

    def initConstantQ(self, q0) :
        """ Initialise the function q with a constant q = q0 """
        def q(x) :
            return q0

        self.q = q

    def initFunctionQ(self, q0, q1, xs, xf) :
        """ Initialise the function q with a step function, with values q0 below xs, 0 above xf,
            and a linear function between xs and xf going from q0 to q1 """
        def q(x) :
            if x < xs :
                return q0
            elif x > xf :
                return 0
            else :
                return q0 + (x - xs) * (q1 - q0) / (xf - xs)

        self.q = q

    def plotQ(self) :
        """ Simple plot of the function q """
        values = [self.q(x) for x in self.x]
        plt.plot(self.x, values)





    # **********************************************************
    #                  Stationary Solutions
    # **********************************************************
    def resolveStationary(self, initialHeight = 1, showPlot = False) :
        """
        Resolving the equation for stationary values
        -initialHeight (Float) : Height of the glacier at x=0
        -showPlot (Boolean) : Boolean for plotting or not the final result
        This function applies a simple forward Euler method, to resolve the following equation :
                                  λ * dh(m+2)/dx = q
        """
        # \\\ Initialisation of the values \\\
        values = [initialHeight]
        self.u = [initialHeight ** (1/(self.m + 2))]

        # \\\ Resolving the PDE \\\
        k = 0
        self.x = [k]
        while values[k] >= 0 and k < 1000000:
            values.append(values[k] + (self.dx / self.l) * self.q(self.dx * k))
            if values[k] >= 0 :
                self.u.append(values[k] ** (1/(self.m+2)))
            else :
                self.u.append(0)

            self.x.append(k * self.dx)
            k = k + 1
        self.u.append(0)
        self.x.append((k+1) * self.dx)
        self.n = len(self.x)

        # \\\ Plotting the result \\\
        if showPlot :
            plt.plot(self.x, self.u)
            title = "Stationary Height of the Glacier\nParameters : m=" + str(self.m) + ", κ=" +str(self.k)
            plt.title(title)
            plt.xlabel("x")
            plt.ylabel("Height of the glacier")


    # **********************************************************
    #                    Compute Flow
    # **********************************************************

    def computeFlow(self, dz, skipParameter = 10, showPlot = False) :
        """ Compute the values of the velocity for previously computed values of h
            - dz : space step along z axis
            - skipParameter : number of skipped vectors in the plot
            - showPlot (Boolean) : argument for plotting the velocity
        """
        c1 = self.k / (self.m + 1)                  # Constants
        c2 = self.k / (self.l * (self.m + 2))      #

        skip = slice(None, None, skipParameter)     # We only compute the vectors we
        indices = range(self.n)[skip]               # want to plot at the end

        for i in  indices :
            u = self.u[i]                           # Value of h at point (x,t)
            x = i * self.dx                         # Value of x
            z = u                                   # Value of z (initialised at the current height)

            # Now we iterate of other some values of x, and calculate each value of
            # the velocity vector starting from the current height, and descending
            # along the z-axis
            while z > 0 :
                vx = c1 * u ** (self.m + 1) * (1 - (1 - z/u) ** (self.m + 1))
                vy = -c2 * ((self.q(x)/u) * (z - \
                (u/(self.m + 1)) * ((1 - z/u) ** (self.m + 1)) - 1))

                self.vx.append(vx)
                self.vy.append(vy)
                self.p.append((vx ** 2 + vy ** 2) ** .5)

                self.coordx.append(i * self.dx)
                self.coordy.append(z)
                z = z - dz

        if showPlot :
            plt.quiver(self.coordx, self.coordy, self.vx, self.vy, self.p, width = 0.002,
                       headwidth = 2.5, headlength = 4, color = "red")


    # **********************************************************
    #                    Compute Perturbations
    # **********************************************************

    def computePerturbation(self, initValue, dt, nbIterations, frames) :
        """ Compute the evolution of a perturbation in time, when a small punctual
            perturbation is introduced at x = 0, of amplitude initValue
            - initValue : Value of the dirac at x = 0
            - dt : time step
            - nbIteration : Number of time iterations
            - frames : the number of time iterations we skip between each frame we plot
        """
        # Quick evaluation of the CFL conditions --------------------
        cfl = dt/self.dx
        print("CFL : 𝛿t/𝛿x = " + str(cfl))
        if cfl > 1 :
            print("Warning : CFL conditions not met")

        # Previously computed stationary state
        h = self.u
        # Initialisation of the perturbation with a dirac at x = 0
        listK = [[initValue]+[0 for i in range(self.n-2)]]

        # We compute the coefficients alpha and beta
        f1 = [self.l * (self.m + 2) * h[i] ** (self.m + 1) for i in range(self.n-1)]
        f2 = [self.l * (self.m + 2) * (self.m + 1) * h[i] ** (self.m + 1) * (h[i+1] - h[i]) / self.dx
              for i in range(self.n -1)]

        alpha = [1 - (self.dx / dt) * f1[i]- dt * f2[i] for i in range(0, self.n - 1)]
        beta = [f1[i] * dt / self.dx for i in range(self.n - 1)]

        for t in range(nbIterations) :
            newK = [initValue]      # The new value at t=t with k(0,t) = initValue
            lastK = listK[-1]       # The last value at t-1

            for x in range(1, self.n - 1) :
                newK.append(alpha[x] * lastK[x] + beta[x] * lastK[x-1])

            listK.append(newK)
            height = [newK[i] + h[i] for i in range(self.n - 1)]+[0]


            # Plotting of the graph
            if (t % frames == 0) :
                plt.plot(self.x, height, label = "t = " + str(t * dt))


    # **********************************************************
    #                    Compute Change
    # **********************************************************

    def computeChange(self, dt, nbIterations, showPlot = False, frames = 100) :
        """ Compute the change of a previously computed glacier,
            after some change have been made on the conditions
            - dt : time step
            - nbIterations : Number of time iterations
            - frames : Number of time steps we skip between two plots

            Returns : A list with all the computed height functions
                      result[i] -> values of the height after i iterations
        """
        # Quick evaluation of the CFL conditions --------------------
        cfl = dt/self.dx
        print("CFL : 𝛿t/𝛿x = " + str(cfl))
        if cfl > 1 :
            print("Warning : CFL conditions not met")

        h0 = self.u                         # Initial height of the glacier
        initialValue = h0[0]                # Value at x = 0 (fixed for all the states of the glacier)
        listH = [h0]
        self.t = [0]

        for t in range(nbIterations) :
            newH = [initialValue]

            for x in range(1, self.n) :
                hi = listH[-1][x]
                him1 = listH[-1][x - 1]
                h = hi - (self.l * dt / self.dx) * (hi ** (self.m + 2) - him1** (self.m + 2)) + \
                            dt * self.q(x * self.dx)
                if h <= 0 : h = 0
                newH.append(h)

            listH.append(newH)
            self.t.append(t * dt)

            # Plotting
            if (showPlot and t%frames == 0 ) :
                plt.plot(self.x[0:self.n], newH, label = "t = " + str(t*dt))

        self.results = listH


    def plotChange2D(self, skipParameter1, skipParameter2) :
        result = self.results
        t, x = np.meshgrid(self.t, self.x)
        lt = len(self.t)
        lx = len(self.x)
        z = np.ones((lt, lx))
        for i in range(lt) :
            for j in range(lx) :
                z[i, j] = result[i][j]

        z = np.transpose(z[0:lt-1, 0:lx-1])
        z2 = np.transpose(z)

        skip1 = slice(None, None, skipParameter1)
        skip2 = slice(None, None, skipParameter2)
        t = t[skip1, skip2]
        x = x[skip1, skip2]
        z = z[skip1, skip2]

        fig = plt.pcolormesh(t, x, z, cmap = "inferno")
        bar = plt.colorbar(fig)
        plt.title("Evolution of the glacier through time")
        plt.ylabel("x")
        plt.xlabel("t")
        plt.tight_layout()
        plt.show()
