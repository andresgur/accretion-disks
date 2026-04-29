from .basedisk import NonAdvectiveDisk
from math import pi
import numpy as np
from scipy.integrate import solve_bvp
from .constants import ccgs, k_T


class ShakuraSunyaevDisk(NonAdvectiveDisk):

    def __init__(self, *args, name="Shakura-Sunyaev Disk", **kwargs):
        super().__init__(*args, name=name, **kwargs)

    def torque(self, R):
        """Analytical expression for the torque when Mass loss is conserved"""
        Rmin = self.CO.Risco * self.Rmin
        return -(
            self.Mdot_0 * self.CO.omega(R) / (2.0 * pi) * (1 - (Rmin / R) ** 0.5)
            - self.Wrphi_in * (Rmin / R) ** 2.0
        )

    def torque_derivative(self, R):
        """Derivative of the Torque"""
        Rmin = self.CO.Risco * self.Rmin
        return -(
            self.Mdot_0 * self.CO.omega(R) / (4 * pi * R) * (4 * (Rmin / R) ** 0.5 - 3)
            + 2.0 * self.Wrphi_in * (Rmin) ** 2 / (R**3)
        )

    def solve(self):
        self.Mdot = self.Mdot_0 * np.ones(self.N)
        self.Wrphi = self.torque(self.R)
        self.H = self.height(self.Wrphi)
        self.Qrad = self.Q_rad(self.H)
        self.Qvis = self.Q_vis(self.Wrphi)
        self.rho = self.density(self.Wrphi, self.H)
        self.vr = self.v_r(self.Mdot_0, self.H, self.rho, self.R)
        self.P = self.pressure(self.H, self.rho)
        self.T = self.temperature(self.P)


class ShakuraSunyaevDiskODE(NonAdvectiveDisk):
    def __init__(self, *args, name="Shakura-Sunyaev Disk", H_in=7e-16, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.H_in = H_in

    def torque(self, R):
        """Analytical expression for the torque when Mass loss is conserved"""
        Rmin = self.CO.Risco
        Omega = self.CO.omega(R)
        return -(
            self.Mdot_0 * Omega / (2.0 * pi) * (1 - (Rmin / R) ** 0.5)
            - self.Wrphi_in * (Rmin / R) ** 2.0
        )

    def torque_derivative(self, R):
        """Derivative of the Torque"""
        Rmin = self.CO.Risco
        Omega = self.CO.omega(R)
        return -(
            self.Mdot_0 * Omega / (4 * pi * R) * (4 * (Rmin / R) ** 0.5 - 3)
            + 2 * self.Wrphi_in * (Rmin) ** 2 / (R**3)
        )

    def Hprime_simplified_2(self, Mdot, H, R, Wrphi, w):
        """Derivative of the height of the disk. Everything in cgs units. Here rho has been replaced and
        the equations have been greatly simplified (mostly for speed purposes)
        Tested against Hprime_simplified and Hprime (units ok) (triple checked)
        Parameters
        ----------
        Mdot: float,
            Mass-accretion rate at the given radius
        H: float,
            Height of the disk
        Wrphi: float
            Stress tensor in the radial and phi coordinates
        dWrphi: float
            Derivative of the stress tensor
        w: float
            Keplerian angular velocity
        """
        return (
            10 * H / (9 * R)
            - np.pi * R * Wrphi / (3 * w * (H + 1e-18) * Mdot)
            - Mdot * w * H / (36 * np.pi * R * (Wrphi + 1e-20))
            - 4 * R * np.pi * ccgs / (9 * Mdot * k_T)
        )

    def Hprime_simplified(self, Mdot, H, R, Wrphi, dWrphi, w):
        """Derivative of the height of the disk. Everything in cgs units. Here rho has been replaced and
        the equations have been greatly simplified (mostly for speed purposes)
        This is Equation 43 from the pdf
        Parameters
        ----------
        Mdot: float,
            Mass-accretion rate at the given radius
        H: float,
            Height of the disk
        Wrphi: float
            Stress tensor in the radial and phi coordinates
        dWrphi: float
            Derivative of the stress tensor
        w: float
            Keplerian angular velocity
        """
        ##print("H NAN", indexes[np.isnan(H)])
        ##print("H 0", indexes[H == 0])
        return (
            1
            / 9
            * (
                12 * H / R
                - 3 * np.pi * R * Wrphi / (w * (H) * Mdot)
                + H * dWrphi / Wrphi
                - 4 * R * np.pi * ccgs / (Mdot * k_T)
            )
        )

    def density(self, Wrphi, H, w):
        """The sign must be flipped to get positive density
        Parameters
        ----------
        Wrphi: float or array-like
            The torque
        H: float
            Scale height
        w: float
            Keplerian angular velocity
        """
        return -Wrphi / (2 * self.alpha * w**2.0 * H**3.0)

    def Hprime(self, Mdot, H, R, Wrphi, dWrphi):
        """Derivative of the height of the disk. Everything in cgs units.
        This gives the same results as Hprime_simplified_2
        Parameters
        ----------
        Mdot: float,
            Mass-accretion rate at the given radius
        H: float,
            Height of the disk
        Wrphi: float
            Stress tensor in the radial and phi coordinates
        dWrphi: float
            Derivative of the stress tensor
        """
        w = self.CO.omega(R)
        rho = self.density(Wrphi, H, w)
        denominator = 9 * rho
        numerator = (
            9 * H * rho / R
            - dWrphi / (2 * self.alpha * H**2 * w**2)
            - 3 / 2 * Wrphi / (self.alpha * H**2 * w**2 * R)
            - 3 * np.pi * R * rho * Wrphi / (Mdot * w * H)
            - 4 * np.pi * R * ccgs * rho / (Mdot * k_T)
        )
        return numerator / denominator

    def ode(self, x, y):
        R = x
        H = y[0]
        Wrphi = self.torque(R)
        w = self.CO.omega(R)
        # Wrphi_in = 0
        # rho = -Wrphi / (2 * self.alpha * w**2.0 * H**3.0)
        # drho = np.gradient(rho, R)
        # dH = self.Hprimesimple(self.Mdot_0, H, Wrphi, rho, drho, R)
        # print(R[0] / self.CO.Risco)
        # print("H is zero", np.where(H == 0))
        # H[0] = 1e-7
        dWrphi = self.torque_derivative(R)
        dH = self.Hprime_simplified(self.Mdot_0, H, R, Wrphi, dWrphi, w)
        # dH = self.Hprime_simplified_2(self.Mdot_0, H, R, Wrphi, w)
        return [dH]

    def bc(self, ya, yb):
        # ya is at the 0 boundary
        # yb at the -1 boundary
        return np.array(
            [
                ya[0] - self.H_in,
                # yb[0]
                # - self.H_in
            ]
        )

    def solve(self, **kwargs):
        R0 = self.CO.Risco
        # 3/4 = 0.75x2 = 1.5
        H_guess = (
            1.5
            * self.CO.Rg
            * self.mdot
            * self.CO.accretion_efficiency(R0)
            * (1 - (R0 / self.R) ** 0.5)
        )
        # self.H_out = H_guess[-1]
        # H_guess[0] = self.H_in  # boundary condition, H=0 at the inner radius

        initial_guess = np.array([H_guess])

        output = solve_bvp(self.ode, self.bc, self.R, initial_guess, **kwargs)

        # Extract solution
        solution = output.sol(self.R)
        self.H = solution[0]
        self.Mdot = self.Mdot_0 * np.ones(self.N)
        self.Wrphi = self.torque(self.R)
        self.Qrad = self.Q_rad(self.H)
        self.Qvis = self.Q_vis(self.Wrphi)
        self.rho = self.density(self.Wrphi, self.H, self.Omega)
        self.vr = self.v_r(self.Mdot_0, self.H, self.rho, self.R)
        self.P = self.pressure(self.H, self.rho)
        self.T = self.temperature(self.P)
