from accretion_disks.shakurasunyaevdisk import ShakuraSunyaevDisk
from .basedisk import AdvectiveDisk, CompositeDisk
from math import pi
from scipy.integrate import solve_bvp
import numpy as np


class ConservativeAdvectiveInnerDisk(AdvectiveDisk):
    def __init__(self, *args, Hout, name="Inner Disk With Outflows", **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.Hout = Hout

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
        # w solve H/Risco for numerical stability, so convert back to H here
        H = y[0] * self.CO.Risco
        Wrphi = self.torque(R)
        w = self.CO.omega(R)
        dWrphi = self.torque_derivative(R)
        dH = self.Hprime_simplified(self.Mdot_0, H, R, Wrphi, dWrphi, w)
        return [dH / self.CO.Risco]  # back to dimensionless

    def bc(self, ya, yb):
        # ya is at the 0 boundary
        # yb at the -1 boundary
        res = ya[0] - self.Hout / self.CO.Risco  # H at the inner boundary is Hout
        return np.array([res])

    def solve(self, **kwargs):
        R0 = self.CO.Risco
        # 3/4 = 0.75x2 = 1.5
        H_guess = (
            1.5
            * self.CO.Rg
            * 1
            * self.CO.accretion_efficiency(R0)
            * (1 - (R0 / self.R) ** 0.5)
        )

        initial_guess = np.array([H_guess])
        # print(np.isnan(initial_guess).any())
        output = solve_bvp(self.ode, self.bc, self.R, initial_guess, **kwargs)

        # Extract solution
        solution = output.sol(self.R)
        self.H = solution[0] * self.CO.Risco
        self.Mdot = self.Mdot_0 * np.ones(self.N)
        self.Wrphi = self.torque(self.R)
        self.Qrad = self.Q_rad(self.H)
        self.Qvis = self.Q_vis(self.Wrphi)
        dWrphi = self.torque_derivative(self.R)

        dH = self.Hprime_simplified(
            self.Mdot_0, self.H, self.R, self.Wrphi, dWrphi, self.Omega
        )
        self.rho = self.density(self.Wrphi, self.H)
        self.Qadv = self.Q_adv(self.Mdot_0, self.H, dH, self.Wrphi, dWrphi)
        self.vr = self.v_r(self.Mdot_0, self.H, self.rho, self.R)
        self.P = self.pressure(self.H, self.rho)
        self.T = self.temperature(self.P)


class ConservativeAdvectiveDisk(CompositeDisk):
    """Base class for disks with different inner and outer solutions.
    Extends NonAdvectiveDisk, but requires innerDiskClass and optionally outerDiskClass (default: ShakuraSunyaevDisk).
    """

    def __init__(
        self,
        innerDiskClass=ConservativeAdvectiveInnerDisk,
        outerDiskClass=ShakuraSunyaevDisk,
        *args,
        name="ConservativeAdvectiveDisk",
        ewind=1,
        **kwargs,
    ):

        super().__init__(
            innerDiskClass,
            outerDiskClass,
            *args,
            name=name,
            ewind=ewind,
            **kwargs,
        )

    def find_Rsph(
        self,
        maxiter=100,
        reltol=1e-4,
        **kwargs,
    ):
        """Finds the radius at which the outer disk luminosity equals the Eddington luminosity,
          which defines the spherization radius Rsph.

        Parameters
        ----------
        maxiter: int, optional
            Maximum number of iterations for the solver.
        reltol: float, optional
            Relative tolerance for convergence.

        Returns
        -------
        Rsph: float
            The calculated spherization radius
        Hrsph: float
            The height of the disk at the spherization radius
        """
        outerDisk = self.outerDiskClass(
            self.CO,
            self.mdot,
            self.alpha,
            Rmin=self.Rmin,
            Rmax=self.Rmax,
            N=self.N,
            name="Outer Disk",
        )
        outerDisk.solve()
        R = np.asarray(outerDisk.R)
        Qrad = np.asarray(outerDisk.Qrad)
        LEdd = outerDisk.CO.LEdd

        # Luminosity released in each annulus.
        dR = np.diff(R)
        dL = 4.0 * np.pi * dR * Qrad[1:] * R[1:]

        # Lout[i] = integrated luminosity from R[i] to R[-1].
        Lout = np.zeros_like(R)
        Lout[:-1] = np.cumsum(dL[::-1])[::-1]

        residual = Lout - LEdd

        if residual[0] < 0:
            raise ValueError(
                "Outer disk is too faint: Rsph extends beyond Rmax or resolution is too low."
            )

        if residual[-1] > 0:
            raise ValueError(
                "Outer boundary is still super-Eddington; increase Rmax to bracket Rsph."
            )

        # Choose Rsph on the original grid, using the radius whose integrated
        # outer luminosity is closest to LEdd.
        idx = np.abs(residual).argmin()
        return R[idx], outerDisk.H[idx]

    def solve(self, **kwargs):

        Rsph, Hout = self.find_Rsph(**kwargs)
        Rsphidx = Rsph / self.CO.Risco

        Ninner = np.count_nonzero(self.R <= Rsph)
        Nouter = self.N - Ninner

        self.innerDisk = self.innerDiskClass(
            self.CO,
            self.mdot,
            self.alpha,
            Rmin=self.Rmin,
            Rmax=Rsphidx,
            N=Ninner,
            name="Inner Disk",
            Hout=Hout,
        )

        self.innerDisk.solve()

        self.outerDisk = self.outerDiskClass(
            self.CO,
            self.mdot,
            self.alpha,
            Rmin=Rsphidx,
            Rmax=self.Rmax,
            N=Nouter,
            name="Outer Disk",
            Wrphi_in=self.innerDisk.Wrphi[-1],
        )

        self.outerDisk.solve()
        self.Rsph = Rsph
        # Combine solutions
        super().solve(**kwargs)
