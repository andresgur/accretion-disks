from .basedisk import NonAdvectiveDisk
from math import pi
from scipy.integrate import solve_bvp
import numpy as np


class InnerDisk(NonAdvectiveDisk):
    def __init__(self, *args, name="Inner Disk With Outflows", ewind=1, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.ewind = ewind

    def torque(self, R):
        """Calculate the torque at a given radius.

        Parameters
        ----------
        R: float or array-like
            The radius at which to calculate the torque.

        Returns
        -------
        float or array-like
            The calculated torque.
        """
        # 2.5 = 5/2
        # 3/2 = 1.5
        Rmin = self.Rmin * self.CO.Risco
        Wrphi = (
            self.Mdot
            * self.Omega
            / (4.0 * np.pi)
            * (1.0 - (R / Rmin) ** 2.5)
            / (1.0 + 1.5 * (R / Rmin) ** 2.5)
        )
        Wrphi[0] = self.Wrphi_in
        return Wrphi

    def Mdot_R(self, R):
        """Calculate the mass-transfer rate at a given radius.

        Parameters
        ----------
        R: float
            The radius at which to calculate the mass-transfer rate.

        Returns
        -------
        float
            The mass-transfer rate at the given radius.
        """
        Rmin = self.Rmin * self.CO.Risco
        Rsph = self.Rmax * self.CO.Risco
        Mdot = (
            self.Mdot_0
            * (Rsph / R) ** 1.5
            * (1 + 1.5 * (R / Rmin) ** 2.5)
            / (1 + 1.5 * (Rsph / Rmin) ** 2.5)
        )
        return Mdot

    def solve(self, **kwargs):
        self.Mdot = self.Mdot_R(self.R)
        self.Wrphi = self.torque(self.R)
        self.H = self.height(self.Wrphi)
        self.Qrad = self.Q_rad(self.H)
        self.Qvis = self.Q_vis(self.Wrphi)
        self.rho = self.density(self.Wrphi, self.H)
        self.vr = self.v_r(self.Mdot, self.H, self.rho, self.R)
        self.P = self.pressure(self.H, self.rho)
        self.T = self.temperature(self.P)


class InnerDiskODE(NonAdvectiveDisk):
    def __init__(self, *args, name="Inner Disk With Outflows", ewind=1, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.ewind = ewind

    def _torque_derivative(self, Mdot, Wrphi, R):
        """Derivative of the torque
        Derived from Equation (4)
        Parameters
        ----------
        Mdot: float,
            Mass-transfer rate at a given radius
        Wrphi: float
            The value of the torque at a given radius
        R: float
            The radius
        w: float
            Keplerian angular velocity at the given radius
        """
        return -(Mdot * self.CO.omega(R) / (4 * pi) + 2 * Wrphi) / R

    def Mdotprime(self, Wrphi, R):
        """From equation 12 and replacing Qrad using Equation 8 and the Pressure using Equation 9
        The derivative of Mdot is positive (it increases as we move outward), because Wrphi is negative, here a negative sign must be kept (it also follows from the derivation)
        All parameters in cgs units
        Parameters
        ----------
        Wrphi:float or array-like,
            The value of the torque at a given radius
        R: float,
            Radius
        """
        return -6.0 * np.pi * Wrphi / (self.CO.omega(R) * R)

    def bc(self, ya, yb):

        return np.array(
            [
                yb[0] - self.Mdot_0,  # Mdot(Rsph) = Mdot_0
                ya[1] - self.Wrphi_in,  # W_rphi(Rmin) = Wrphi_in
            ]
        )

    def ode(self, x, y):
        """ODE system for BVP solver with parameter p=[Rsph]"""
        R = x
        Mdot = y[0]
        Wrphi = y[1]

        dMdot_dR = self.Mdotprime(Wrphi, R)
        dWrphi_dR = self._torque_derivative(Mdot, Wrphi, R)

        # Convert to derivatives wrt x and return as 2D array
        return np.vstack([dMdot_dR, dWrphi_dR])

    def solve(self, **kwargs):
        """Solve the coupled ODEs with BVP solver and integral constraint."""

        # Initial guess for [Mdot, Wrphi]
        Rsph = self.Rmax * self.CO.Risco
        Rmin = self.Rmin * self.CO.Risco
        Mdot_guess = self.R / Rsph * self.Mdot_0
        Wrphi_guess = (
            -self.Mdot_0
            * self.R
            / (Rsph * Rmin)
            * self.Omega
            / (4.0 * np.pi)
            * (1 - (self.R / Rmin) ** (2.5))
            / (1 + 1.5 * (self.R / Rmin) ** (2.5))
        )
        Wrphi_guess[0] = self.Wrphi_in

        initial_guess = np.array([Mdot_guess, Wrphi_guess])

        # Solve BVP with parameter [Rsph]
        output = solve_bvp(self.ode, self.bc, self.R, initial_guess, **kwargs)

        # Extract solution
        solution = output.sol(self.R)
        self.Mdot = solution[0]
        self.Wrphi = solution[1]
        self.H = self.height(self.Wrphi)
        self.Qrad = self.Q_rad(self.H)
        self.Qvis = self.Q_vis(self.Wrphi)
        self.rho = self.density(self.Wrphi, self.H)
        self.vr = self.v_r(self.Mdot, self.H, self.rho, self.R)
        self.P = self.pressure(self.H, self.rho)
        self.T = self.temperature(self.P)
