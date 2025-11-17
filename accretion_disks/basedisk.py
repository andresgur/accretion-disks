from abc import ABC, abstractmethod
from .constants import M_suncgs, k_T, ccgs
import numpy as np
from math import pi


class Disk(ABC):
    def __init__(self, CO, mdot, alpha=0.1, name="disk", Rmin=1, Rmax=500, N=20000):
        self.CO = CO
        self.mdot = mdot
        self.alpha = alpha
        self.Mdot_0 = self.CO.Medd * self.mdot
        self.name = name
        print("Disk %s with M = %.1f M_sun, dot(m) = %.1f and alpha = %.1f and spin = %.1f" % (self.name, self.CO.M / M_suncgs,
                      self.mdot, self.alpha, self.CO.a))
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.R = np.linspace(self.Rmin, self.Rmax, N) * self.CO.Risco
        self.Omega = self.CO.omega(self.R)

  
    def density(self, Wrphi, H):
        """The sign must be flipped to get positive density
        Parameters
        ----------
        Wrphi: float
            The torque
        H: float
            Scale height
        w: float
            Keplerian angular velocity
        """
        return -Wrphi / (2 * self.alpha * self.Omega**2. * H**3.)


    def Q_rad(self, H):
        """Radiative energy per unit surface. All quantities in cgs
        Parameters
        ----------
        R: float
            Radius
        """
        Qrad = H * self.Omega**2. *ccgs /k_T
        return Qrad # checked
    

    def v_r(self, Mdot, H, rho, R):
        """Radial velocity. Equation 13 from Lipunova+1999
        Parameters
        ----------
        Mdot: float
            Mass accretion rate in g/s at each radii
        
        """
        return Mdot / (rho * H * 4 * pi * R)
    
    @abstractmethod
    def solve():
        pass
