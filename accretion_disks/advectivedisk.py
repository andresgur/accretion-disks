from .basedisk import AdvectiveDisk
from math import pi
from scipy.integrate import solve_bvp
import numpy as np


class ConservativeInnerDisk(AdvectiveDisk):
    def __init__(self, *args, name="Inner Disk With Outflows", H_in=0, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.H_in = H_in
        self.solve()

    def torque(self, R):
        """Analytical expression for the torque when Mass loss is conserved"""
        Rmin = self.CO.Risco
        return -(
            self.Mdot_0 * self.CO.omega(R) / (2.0 * pi) * (1 - (Rmin / R) ** 0.5)
            - self.Wrphi_in * (Rmin / R) ** 2.0
        )

    def torque_derivative(self, R):
        """Derivative of the Torque"""
        Rmin = self.CO.Risco
        return -(
            self.Mdot_0 * self.CO.omega(R) / (4 * pi * R) * (4 * (Rmin / R) ** 0.5 - 3)
            + 2.0 * self.Wrphi_in * (Rmin) ** 2 / (R**3)
        )

    def ode(self, x, y):
        R = x
        H = y[0]
        Wrphi = self.torque(R)
        w = self.CO.omega(R)
        # rho = -Wrphi / (2 * self.alpha * w**2.0 * H**3.0)
        # drho = np.gradient(rho, R)
        # dH = self.Hprimesimple(self.Mdot_0, H, Wrphi, rho, drho, R)
        # print(R[0] / self.CO.Risco)
        # print("H is zero", np.where(H == 0))
        # H[0] = 1e-7
        dWrphi = self.torque_derivative(R)
        # print("Torque is zero", np.where(Wrphi == 0))
        dH = self.Hprime_simplified(self.Mdot_0, H, R, Wrphi, dWrphi, w)
        # dH = self.height_derivative_noQadv(self.Mdot_0, H, Wrphi, R)
        return [dH]

    def bc(self, ya, yb):
        # ya is at the 0 boundary
        # yb at the -1 boundary
        return np.array(
            [
                ya[0] - self.H_in,
                # yb[0]
                # - self.H_out
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

        self.H_out = H_guess[-1]
        # H_guess[0] = self.H_in  # boundary condition, H=0 at the inner radius

        initial_guess = np.array([H_guess])
        output = solve_bvp(
            self.ode,
            self.bc,
            self.R,
            initial_guess,
            verbose=2,
            max_nodes=100000000,
            tol=0.01,
        )

        # Extract solution
        solution = output.sol(self.R)
        self.H = solution[0]
        self.Mdot = self.Mdot_0 * np.ones(self.N)
        self.Wrphi = self.torque(self.R)
        self.Qrad = self.Q_rad(self.H)
        self.Qvis = self.Q_vis(self.Wrphi)

        dH = self.Hprime_simplified(
            self.Mdot_0, self.H, self.R, self.Wrphi, dWrphi, self.Omega
        )
        self.rho = self.density(self.Wrphi, self.H)
        dWrphi = self.torque_derivative(self.R)
        drho = self.density_derivative(self.Wrphi, dWrphi, self.H, dH)
        self.Qadv = self.Q_adv(self.Mdot_0, self.H, dH, self.rho, drho)
        self.vr = self.v_r(self.Mdot_0, self.H, self.rho, self.R)
