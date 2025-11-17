from .basedisk import Disk
from math import pi
from .constants import ccgs, k_T

class ShakuraSunyaevDisk(Disk):

    def __init__(self, *args, name="Shakura-Sunyaev Disk", torque_Rmin=0., **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.torque_Rmin = torque_Rmin  # Torque at the inner boundary

    def height(self, Wrphi):
        """
        Disk height. Inputs in cgs units

        Parameters
        ----------
        """
        # 3/4 = 0.75
        H = - 0.75 * k_T * Wrphi / (self.Omega * ccgs)
        return H
    

    def torque(self, R):
        """Analytical expression for the torque when Mass loss is conserved
        """
        Rmin = self.Rmin * self.CO.Risco
        return -(self.Mdot_0 * self.Omega / (2. * pi) * (1 - (Rmin / R)**0.5) + self.torque_Rmin * (Rmin / R)**2.)
    

    def torque_derivative(self, R):
        """Derivative of the Torque"""
        Rmin = self.Rmin * self.CO.Risco
        return -(self.Mdot_0 * self.Omega / (4 * pi * R) * (4 * (Rmin / R)**0.5 - 3) - 2 * self.torque_Rmin * (Rmin)**2 / (R**3))


    def solve(self):
        self.Wrphi = self.torque(self.R)
        self.H = self.height(self.Wrphi)
        self.Qrad = self.Q_rad(self.H)
        self.rho = self.density(self.Wrphi, self.H)
        self.vr = self.v_r(self.Mdot_0, self.H, self.rho, self.R)