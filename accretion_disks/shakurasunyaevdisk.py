from .basedisk import NonAdvectiveDisk
from math import pi
import numpy as np

class ShakuraSunyaevDisk(NonAdvectiveDisk):

    def __init__(self, *args, name="Shakura-Sunyaev Disk", **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.solve()

    def torque(self, R):
        """Analytical expression for the torque when Mass loss is conserved
        """
        Rmin = self.Rmin * self.CO.Risco
        return -(self.Mdot_0 * self.Omega / (2. * pi) * (1 - (Rmin / R)**0.5) - self.Wrphi_in * (Rmin / R)**2.)
    

    def torque_derivative(self, R):
        """Derivative of the Torque"""
        Rmin = self.Rmin * self.CO.Risco
        return -(self.Mdot_0 * self.Omega / (4 * pi * R) * (4 * (Rmin / R)**0.5 - 3) - 2 * self.Wrphi_in * (Rmin)**2 / (R**3))


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