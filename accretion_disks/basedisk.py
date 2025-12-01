from abc import ABC, abstractmethod
from .constants import M_suncgs, k_T, ccgs, A
import numpy as np
from math import pi
import matplotlib.pyplot as plt


class Disk(ABC):
    def __init__(
        self, CO, mdot, alpha=0.1, name="disk", Rmin=1, Rmax=1e5, N=20000, Wrphi_in=-0.1
    ):
        self._CO = CO
        self._mdot = mdot
        if alpha >= 1 or alpha < 0:
            raise ValueError("Alpha must be between 0 and 1!", f"alpha = {alpha:.2f}")
        self._alpha = alpha
        self.name = name
        self.Rmin = Rmin
        self.Rmax = Rmax
        if self.Rmin < 1:
            raise ValueError("Rmin must be greater than 1!", f"Rmin = {Rmin:.1f}")
        if self.Rmin >= self.Rmax:
            raise ValueError("Rmin must be smaller than Rmax")
        self.R = np.geomspace(self.Rmin, self.Rmax, N) * self.CO.Risco
        self.N = N
        self.Mdot_0 = self.CO.MEdd * self.mdot
        self.Omega = self.CO.omega(self.R)

        if Wrphi_in > 0:
            raise ValueError(
                "Torque is negatively defined! Inner torque must be of negative sign"
            )

        self.Wrphi_in = Wrphi_in

    def __repr__(self):
        return (
            "Disk %s with M = %.1f M_sun, dot(m) = %.1f and alpha = %.1f and spin = %.1f and N = %d datapoints"
            % (
                self.name,
                self.CO.M / M_suncgs,
                self.mdot,
                self.alpha,
                self.CO.a,
                self.N,
            )
        )

    @property
    def mdot(self) -> float:
        return self._mdot

    @mdot.setter
    def mdot(self, value: float):
        self._mdot = value
        self.Mdot_0 = self.CO.MEdd * self.mdot
        self.solve()

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        """Alpha only affects disk structure quantities, so we do not need to recompute the energy balance equations"""
        if value >= 1 or value < 0:
            raise ValueError("Alpha must be between 0 and 1!", f"alpha = {value:.2f}")
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
        return -Wrphi / (2 * self.alpha * self.Omega**2.0 * H**3.0)

    def temperature(self, P):
        """
        Parameters
        ----------
        P: float or array-like
            The pressure at each radii in cgs units

        """
        T = (3 * P / A) ** (1 / 4)
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
        return self.Omega**2.0 * H**2 * rho

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
        Qrad = H * self.Omega**2.0 * ccgs / k_T
        return Qrad  # checked

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

    def plot(
        self,
    ):
        deltaR = np.diff(self.R)
        lum_cumsum = 4.0 * np.pi * np.cumsum(deltaR * self.Qrad[1:] * self.R[1:])
        # For each R, get the cumulative luminosity for R > R[i]
        lums = lum_cumsum[-1] - lum_cumsum

        fig, axes = plt.subplots(
            5, sharex=True, figsize=(12, 8), gridspec_kw={"hspace": 0.1}
        )
        axes[0].set_xscale("log")
        axes[0].set_xlim(self.Rmin, self.Rmax)

        r = self.R / self.CO.Risco
        axes[0].plot(r, self.H / self.R, label="H / R")
        # axes[0].plot(r[1:], lums / self.CO.LEdd, label=r"$L(r > R)$", ls="--")
        # axes[0].plot(
        #   r, self.Mdot / self.Mdot_0, label=r"$\dot{M}(r) / \dot{M}_0$", ls=":"
        # )
        axes[0].legend()
        axes[0].set_ylim(bottom=0)

        axes[1].plot(r, -self.Wrphi, label=r"$W_\mathrm{r\phi}$")
        axes[1].set_yscale("log")
        axes[1].set_ylabel(r"$W_\mathrm{r\phi}$")

        axes[2].plot(
            r, self.Qrad / self.Qvis, label=r"$Q_\mathrm{rad}/ Q_\mathrm{vis}$"
        )
        axes[2].plot(
            r, self.Qadv / self.Qvis, label=r"$Q_\mathrm{adv}/ Q_\mathrm{vis}$"
        )
        axes[2].legend()

        axes[3].plot(r[1:], self.rho[1:])
        axes[3].set_ylabel(r"$\rho$ (g/cm$^3$)")
        axes[3].set_yscale("log")

        axes[4].plot(r, self.vr / ccgs)
        axes[4].set_ylabel(r"$v/c$")

        axes[-1].set_xlabel(r"$R / R_\mathrm{isco}$")
        return fig, axes


class NonAdvectiveDisk(Disk):
    """Base class for non-advective disks"""

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
        H = -0.75 * k_T * Wrphi / (self.Omega * ccgs)

        return H


class AdvectiveDisk(Disk):
    def __init__(self, *args, name="AdvectiveDisk", **kwargs):
        super().__init__(*args, name=name, **kwargs)

    def Q_adv(self, Mdot, H, dH, rho, drho):
        w = self.Omega
        factor = 6 * dH * rho - H * drho - 9 * H * rho / self.R
        return Mdot * w**2 * H / (4 * np.pi * self.R * rho) * factor

    def height_derivative(self, Mdot, H, Wrphi, dWrphi, R):
        """Derivative of the height of the disk. Everything in cgs units. Here rho has been replaced and
        the equations have been greatly simplified (mostly for speed purposes)
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
        # Here we need to keep the omega(R) explicit as the R grid changes during the solver
        omega = self.CO.omega(R)
        return (
            1
            / 9.0
            * (
                12.0 * H / R
                - 3 * np.pi * R * Wrphi / (omega * H * Mdot)
                + H * dWrphi / Wrphi
                - 4 * R * np.pi * ccgs / (Mdot * k_T)
            )
        )

    def Hprimesimple(self, Mdot, H, Wrphi, rho, drho, R):
        """Equation 35 from the pdf"""

        omega = self.CO.omega(R)
        factor = 1 / (6.0 * rho)
        dH = factor * (
            -3.0 * Wrphi * np.pi * R * rho / (Mdot * omega * H)
            - 4.0 * np.pi * ccgs * R * rho / (Mdot * k_T)
            + H * drho
            + 9 * H * rho / R
        )
        # 9 * H * rho / R
        # + H * drho
        return dH

    def Hprime(self, Mdot, H, Wrphi, dWrphi):
        """Derivative of the height of the disk. Everything in cgs units.
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
        w = self.Omega
        rho = self.density(Wrphi, H)
        denominator = 6.0 * rho - 1.5 * Wrphi / (self.alpha * H**3 * w**2)
        numerator = (
            9.0 * H * rho / self.R
            - dWrphi / (2.0 * self.alpha * H**2 * w**2)
            - 3 / 2 * Wrphi / (self.alpha * H**2 * w**2 * self.R)
            - 3 * np.pi * self.R * rho * Wrphi / (Mdot * w * H)
            - 4 * np.pi * self.R * ccgs * rho / (Mdot * k_T)
        )
        return numerator / denominator

    def Hprime_simplified(self, Mdot, H, R, Wrphi, dWrphi, w):
        """Derivative of the height of the disk. Everything in cgs units. Here rho has been replaced and
        the equations have been greatly simplified (mostly for speed purposes)
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
            1
            / 9
            * (
                12 * H / R
                - 3 * np.pi * R * Wrphi / (w * H * Mdot)
                + H * dWrphi / Wrphi
                - 4 * R * np.pi * ccgs / (Mdot * k_T)
            )
        )

    def height_derivative2(self, Mdot, H, Wrphi, R):
        omega = self.CO.omega(R)
        dH = (
            10.0 / 9.0 * H / R
            - np.pi * R * Wrphi / (3.0 * omega * H * Mdot)
            - Mdot * omega * H / (36.0 * np.pi * R * Wrphi)
            - 4 * np.pi * R * ccgs / (9.0 * Mdot * k_T)
        )
        return dH

    def height_derivative_noQadv(self, Mdot, H, Wrphi, R):
        omega = self.CO.omega(R)
        return (
            1
            / 9.0
            * (
                -3 * np.pi * R * Wrphi / (omega * H * Mdot)
                - 4 * R * np.pi * ccgs / (Mdot * k_T)
            )
        )

    def density_derivative(self, Wrphi, dWrphi, H, dH):
        """Derivative of the density (unused for now) at the given radius
        Parameters
        ----------
        Wrphi: float,
            Stress tensor
        dWrphi: float,
            Derivative of the stress tensor
        H: float,
            Height of the disk
        dH: float
            Derivative of the height of the disk at the given radius
        R: float,
            Radius
        """
        w = self.Omega
        return (
            -1
            / (2 * self.alpha * H**3 * w**2)
            * (dWrphi - 3 * Wrphi * dH / H + 3 * Wrphi / self.R)
        )


class CompositeDisk(Disk):
    """Base class for disks with different inner and outer solutions.
    Extends NonAdvectiveDisk, but requires innerDiskClass and optionally outerDiskClass (default: ShakuraSunyaevDisk).
    """

    def __init__(
        self,
        innerDiskClass,
        outerDiskClass,
        *args,
        name="CompositeDisk",
        ewind=1,
        **kwargs,
    ):

        super().__init__(*args, name=name, **kwargs)
        self.innerDiskClass = innerDiskClass
        self.outerDiskClass = outerDiskClass
        self.ewind = ewind
        self.solve()

    def adjust_Rsph(
        self,
        maxiter=100,
        reltol=1e-4,
    ):
        """Calculate the spherization radius based on the radiative flux.

        Parameters
        ----------
        maxiter: int, optional
            Maximum number of iterations for the solver.
        reltol: float, optional
            Relative tolerance for convergence.

        Returns
        -------
        float or None
            The calculated spherization radius or None if not found.
        innerDisk: InnerDisk
            The solved inner disk object whithin Rsph
        outerDisk: ShakuraSunyaevDisk
            The solved outer disk object, which is a SS73 disk with modified boundary conditions.
        """
        Ra = self.R[0] // self.CO.Risco
        Rb = self.R[-1] // self.CO.Risco

        outerdisk = self.outerDiskClass(
            self.CO,
            self.mdot,
            self.alpha,
            Rmax=self.Rmax,
            Rmin=self.Rmin,
            N=self.N,
            name="Temporary SS Disk",
        )
        L_Ra = outerdisk.L() - self.CO.LEdd
        if L_Ra < 0:
            raise ValueError(
                "Outer disk is either too short (and Rsph extends beyond Rmax) or there are too few datapoints!"
                + "Increase the number of datapoints or the maximum radius of the calculation!"
            )

        for i in range(maxiter):
            R_c = (Ra + Rb) // 2.0
            Ninner = (self.R <= R_c * self.CO.Risco).sum()
            # reset the maximum radius
            innerDisk = self.innerDiskClass(
                self.CO,
                self.mdot,
                self.alpha,
                Rmin=self.Rmin,
                Rmax=R_c,
                N=Ninner,
                name="Inner Disk",
                Wrphi_in=self.Wrphi_in,
                ewind=self.ewind,
            )
            innerDisk.solve()
            Nouter = self.N - Ninner
            if Nouter == 0:
                raise ValueError(
                    "Run out of points in Rsph calculation. Increase the number of grid points!"
                )
            # create truncated SS73 with new Wrphi boundary condition
            outerDisk = self.outerDiskClass(
                self.CO,
                self.mdot,
                self.alpha,
                Rmin=R_c,
                Rmax=self.Rmax,
                N=Nouter,
                name="Outer Disk",
                Wrphi_in=innerDisk.Wrphi[-1],
            )
            outerDisk.solve()
            L_Rc = outerDisk.L() - self.CO.LEdd
            err = abs(R_c - Ra)
            if err / R_c < reltol or L_Rc == 0:
                return R_c, innerDisk, outerDisk
            if (L_Rc * L_Ra) < 0:
                Rb = R_c
            else:
                Ra = R_c
                L_Ra = L_Rc
        return None

    @Disk.mdot.setter
    def mdot(self, value):
        self._mdot = value
        self.Mdot_0 = self.CO.MEdd * self.mdot
        self.innerDisk.mdot = value
        self.outerDisk.mdot = value
        self.solve()

    @Disk.alpha.setter
    def alpha(self, value):
        self._alpha = value
        self.innerDisk.alpha = value
        self.outerDisk.alpha = value
        # recalculate only variables that depend on alpha
        self.rho = np.concatenate((self.innerDisk.rho, self.outerDisk.rho))
        self.vr = np.concatenate((self.innerDisk.vr, self.outerDisk.vr))
        self.P = np.concatenate((self.innerDisk.P, self.outerDisk.P))
        self.T = np.concatenate((self.innerDisk.T, self.outerDisk.T))

    @Disk.CO.setter
    def CO(self, value):
        self._CO = value
        self.Omega = self.CO.omega(self.R)
        self.Mdot_0 = self.CO.MEdd * self.mdot
        self.innerDisk.CO = value
        self.outerDisk.CO = value
        self.solve()

    def solve(self):

        Rsph, self.innerDisk, self.outerDisk = self.adjust_Rsph()
        self.Rsph = Rsph * self.CO.Risco
        # Combine solutions
        self.Mdot = np.concatenate((self.innerDisk.Mdot, self.outerDisk.Mdot))
        self.Wrphi = np.concatenate((self.innerDisk.Wrphi, self.outerDisk.Wrphi))
        self.H = np.concatenate((self.innerDisk.H, self.outerDisk.H))
        self.Qrad = np.concatenate((self.innerDisk.Qrad, self.outerDisk.Qrad))
        self.Qadv = np.concatenate((self.innerDisk.Qadv, self.outerDisk.Qadv))
        self.Qvis = np.concatenate((self.innerDisk.Qvis, self.outerDisk.Qvis))
        self.rho = np.concatenate((self.innerDisk.rho, self.outerDisk.rho))
        self.vr = np.concatenate((self.innerDisk.vr, self.outerDisk.vr))
        self.P = np.concatenate((self.innerDisk.vr, self.outerDisk.P))
        self.T = np.concatenate((self.innerDisk.T, self.outerDisk.T))

    def plot(self):
        fig, axes = super().plot()
        for ax in axes:
            ax.axvline(self.Rsph / self.CO.Risco, color="black", ls="--", label="Rsph")

        axes[1].legend()
        return fig, axes
