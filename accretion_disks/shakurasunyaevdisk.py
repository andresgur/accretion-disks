from basedisk import Disk
from math import pi
from constants import ccgs, k_T

class ShakuraSunyaevDisk(Disk):

    def __init__(self, CO, mdot, alpha=0.1, name="Shakura-Sunyaev Disk", Rmin=1, Rmax=500, N=1000, torque_Rmin=0):
        super().__init__(CO, mdot, alpha, name, Rmin, Rmax, N)
        self.torque_Rmin = 0  # Torque at the inner boundary

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
        and assuming the torque at the inner boundary is 0
        """
        Rmin = R[0]
        return -(self.Mdot_0 * self.Omega / (2. * pi) * (1 - (Rmin / R)**0.5) + self.torque_Rmin * (Rmin / R)**2.)


    def solve(self):
        self.Wrphi = self.torque(self.R)
        self.H = self.height(self.Wrphi)
        self.Qrad = self.Q_rad(self.H)
        self.rho = self.density(self.Wrphi, self.H)
        self.vr = self.v_r(self.Mdot_0, self.H, self.rho, self.R)