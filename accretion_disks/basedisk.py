from abc import ABC, abstractmethod
from .constants import M_suncgs, k_T, ccgs, A
import numpy as np
from math import pi
import matplotlib.pyplot as plt


class Disk(ABC):
    def __init__(self, CO, mdot, alpha=0.1, name="disk", Rmin=1, Rmax=1e5, N=20000, Wrphi_in=-0.1):
        self._CO = CO
        self._mdot = mdot
        if alpha>=1 or alpha <0:
            raise ValueError("Alpha must be between 0 and 1!", f"alpha = {alpha:.2f}")
        self._alpha = alpha
        self.name = name
        self.Rmin = Rmin
        self.Rmax = Rmax
        if self.Rmin <1:
            raise ValueError("Rmin must be greater than 1!", f"Rmin = {Rmin:.1f}")
        if self.Rmin >= self.Rmax:
            raise ValueError("Rmin must be smaller than Rmax")
        self.R = np.geomspace(self.Rmin, self.Rmax, N) * self.CO.Risco
        self.N = N
        self.Mdot_0 = self.CO.MEdd * self.mdot
        self.Omega = self.CO.omega(self.R)

        if Wrphi_in >0:
            raise ValueError("Torque is negatively defined! Inner torque must be of negative sign")
        
        self.Wrphi_in = Wrphi_in    

    def __repr__(self):
        return "Disk %s with M = %.1f M_sun, dot(m) = %.1f and alpha = %.1f and spin = %.1f and N = %d datapoints" % (self.name, self.CO.M / M_suncgs,
                      self.mdot, self.alpha, self.CO.a, self.N)
    
    @property
    def mdot(self) -> float:
        return self._mdot

    @mdot.setter
    def mdot(self, value: float):
        self._mdot = value
        self.Mdot_0 = self.CO.MEdd * self.mdot
        self.solve()

    @property
    def alpha(self) ->float:
        return self._alpha
    
    @alpha.setter
    def alpha(self, value: float):
        """Alpha only affects disk structure quantities, so we do not need to recompute the energy balance equations"""
        if value>=1 or value <0:
            raise ValueError("Alpha must be between 0 and 1!", f"alpha = {alpha:.2f}")
        oldalpha = self.alpha
        self._alpha = value
        # rho is \propto 1 / \alpha
        self.rho = self.rho * oldalpha / self._alpha
        # vr is \propto 1 / rho -> \propto \alpha
        self.vr = self.vr * self._alpha / oldalpha
        # P is \propto \rho -> \propto 1 / \alpha
        self.P = self.P * oldalpha / self._alpha
        self.T = self.temperature(self.P)


    @property
    def CO(self):
        return self._CO

    @CO.setter
    def CO(self, value):
        self._CO = value
        self.R = np.geomspace(self.Rmin, self.Rmax, self.N) * self.CO.Risco
        self.Omega = self.CO.omega(self.R)
        self.Mdot_0 = self.CO.MEdd * self.mdot
        self.solve()
    
    
    def density(self, Wrphi, H):
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
        return -Wrphi / (2 * self.alpha * self.Omega**2. * H**3.)
    
    def temperature(self, P):
        """
        Parameters
        ----------
        P: float or array-like
            The pressure at each radii in cgs units
        
        """
        T = (3 * P / A)** (1/4)
        return T
    

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
    
    def v_r(self, Mdot, H, rho, R):
        """Radial velocity. Equation 13 from Lipunova+1999
        Parameters
        ----------
        Mdot: float
            Mass accretion rate in g/s at each radii
        
        """
        return Mdot / (rho * H * 4 * pi * R)
    

    def Q_vis(self, Wrphi):
        """Returns the viscosity or total energy release in per disk annuli at each radii
        Parameters
        ---------
        Wrphi: float or array-like
            The torque at each radii

        Returns
        ------
        Qvis: float or array-like
            The energy released
        """
        Qvis = -0.75 * self.Omega * Wrphi
        return Qvis
    
    def Q_rad(self, H):
        """Radiative energy per unit surface. All quantities in cgs
        Parameters
        ----------
        R: float
            Radius
        """
        Qrad = H * self.Omega**2. *ccgs /k_T
        return Qrad # checked
    
    
    def L(self, Rmin=None, Rmax=None):
        deltaR = np.diff(self.R)
        if Rmin is None:
            Rmin = self.Rmin * self.CO.Risco
        if Rmax is None:
            Rmax = self.Rmax * self.CO.Risco
        # Fix: select radii between Rmin and Rmax
        R_range = (self.R[1:] >= Rmin) & (self.R[1:] <= Rmax)
        L = 4 * pi * (self.R[1:] * self.Qrad[1:] * deltaR)[R_range].sum()
        return L
    


    @abstractmethod
    def solve():
        pass


    def plot(self,):
        deltaR = np.diff(self.R)
        lum_cumsum = 4. * np.pi *  np.cumsum(deltaR * self.Qrad[1:] * self.R[1:])
        # For each R, get the cumulative luminosity for R > R[i]
        lums = lum_cumsum[-1] - lum_cumsum

        fig, axes = plt.subplots(5, sharex=True, figsize=(12, 8), gridspec_kw={"hspace":0.1})
        axes[0].set_xscale("log")
        axes[0].set_xlim(self.Rmin, self.Rmax)

        r = self.R / self.CO.Risco
        axes[0].plot(r, self.H / self.R, label="H / R")
        axes[0].plot(r[1:], lums / self.CO.LEdd, label=r"$L(r > R)$", ls="--")
        axes[0].plot(r, self.Mdot / self.Mdot_0, label=r"$\dot{M}(r) / \dot{M}_0$", ls=":")
        axes[0].legend()
        axes[0].set_ylim(bottom=0)

        axes[1].plot(r, -self.Wrphi, label=r"$W_\mathrm{r\phi}$")
        axes[1].set_yscale("log")
        axes[1].set_ylabel(r"$W_\mathrm{r\phi}$")

        axes[2].plot(r, self.Qrad / self.Qvis, label=r"$Q_\mathrm{rad}/ Q_\mathrm{vis}$")
        axes[2].plot(r, self.Qadv / self.Qvis, label=r"$Q_\mathrm{adv}/ Q_\mathrm{vis}$")
        axes[2].legend()

        axes[3].plot(r[1:], self.rho[1:])
        axes[3].set_ylabel(r"$\rho$ (g/cm$^3$)")
        axes[3].set_yscale("log")


        axes[4].plot(r, self.vr / ccgs)
        axes[4].set_ylabel(r"$v/c$")

        axes[-1].set_xlabel(r"$R / R_\mathrm{isco}$")
        return fig, axes
    

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
    