from abc import ABC, abstractmethod
from .constants import M_suncgs, k_T, ccgs
import numpy as np
from math import pi


class Disk(ABC):
    def __init__(self, CO, mdot, alpha=0.1, name="disk", Rmin=1, Rmax=500, N=20000):
        self.CO = CO
        self.mdot = mdot
        self.alpha = alpha
        self.Mdot_0 = self.CO.MEdd * self.mdot
        self.name = name
        print("Disk %s with M = %.1f M_sun, dot(m) = %.1f and alpha = %.1f and spin = %.1f and N = %d datapoints" % (self.name, self.CO.M / M_suncgs,
                      self.mdot, self.alpha, self.CO.a, N))
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.R = np.linspace(self.Rmin, self.Rmax, N) * self.CO.Risco
        self.N = N
        self.Omega = self.CO.omega(self.R)
        self.Mdot = self.Mdot_0 * np.ones(self.N)


  
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
    

    def pressure(self, H, rho):
        """Returns the (radiation) pressure of the disk
        Parameters
        ----------
        H: float
            Height
        rho: float
            Density
        """
        return self.Omega**2. * H**2 * rho


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
    
    def L(self, Rmin=None, Rmax=None):
        deltaR = self.R[1] - self.R[0]
        if Rmin is None:
            Rmin = self.Rmin * self.CO.Risco
        if Rmax is None:
            Rmax = self.Rmax * self.CO.Risco
        # Fix: select radii between Rmin and Rmax
        R_range = (self.R >= Rmin) & (self.R <= Rmax)
        L = 4 * pi * (self.R[R_range] * self.Qrad[R_range]).sum() * deltaR
        return L

    @abstractmethod
    def solve():
        pass
class NonAdvectiveDisk(Disk):
    """Base class for non-advective disks
    """

    def __init__(self, *args, name="NonAdvectiveDisk", **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.Qadv = np.zeros_like(self.R)
        
    def height(self, Wrphi):
        """
        Disk height. Inputs in cgs units. From Equation 13 in the pdf

        Parameters
        ----------
        Wrphi: float or array-like
            The torque at each radius in cgs units
        """
        # 3/4 = 0.75
        H = - 0.75 * k_T * Wrphi / (self.Omega * ccgs)
        
        return H
