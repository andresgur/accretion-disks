from .basedisk import NonAdvectiveDisk
from .shakurasunyaevdisk import ShakuraSunyaevDisk
from math import pi
from .constants import A, Gcgs
from scipy.integrate import solve_bvp
import numpy as np



def mass_transfer_inner_radius(m_0, e_wind=0.5):
    """Calculate the mass-transfer rate at the inner radius of the disk.

    Parameters
    ----------
    m_0: float
        Mass-transfer rate at the donor.
    e_wind: float
        Fraction of radiative energy that goes to accelerate the outflow.

    Returns
    -------
    float
        Mass-transfer rate at the inner radius of the disk in units of m_0.
    """
    a = e_wind * (0.83 - 0.25 * e_wind)

    return (1 - a) / (1 - a * (2/5 * m_0) ** (- 0.5) )


class DiskWithOutflowsRemove(NonAdvectiveDisk):
    def __init__(self, *args, name="Disk With Outflows", Wrphi_in=0, ewind=1, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.Wrphi_in = Wrphi_in
        self.ewind = ewind
        self.Rsph = 1.62 * self.mdot * self.CO.Risco
        self.Mdot = self.Mdot_R(self.R)
        self.Ledd = self.CO.LEdd
        
        

    def get_spherization_radius(self, Qrad, maxiter=100, reltol=1e-8):
        """Calculate the spherization radius based on the radiative flux.

        Parameters
        ----------
        Qrad: array-like
            Radiative flux at different radii.
        maxiter: int, optional
            Maximum number of iterations for the solver.
        reltol: float, optional
            Relative tolerance for convergence.

        Returns
        -------
        float or None
            The calculated spherization radius or None if not found.
        """
        Ra = self.R[0]
        Rb = self.R[-1]
        deltaR = self.R[1] - self.R[0]
        R_range = (self.R > Ra)
        fourpideltaR = 4 * np.pi * deltaR
        QradR = Qrad * self.R
        L_Ra = fourpideltaR * QradR[R_range].sum() - self.CO.LEdd

        for i in range(maxiter):
            R_c = (Ra + Rb) / 2.0
            # integrate outer radii
            R_range = (self.R > R_c)
            L_Rc = fourpideltaR * QradR[R_range].sum() - self.CO.LEdd
            err = abs(R_c - Ra)
            if err / R_c < reltol or L_Rc == 0:
                print(f"Solution found after {i+1} iterations with error {err:.2e}")
                return R_c
            if (L_Rc * L_Ra) < 0:
                Rb = R_c
            else:
                Ra = R_c
                L_Ra = L_Rc
        return None
    
    def outerTorque(self, R, Wrphin, Rmin):    

        Wphi = -(self.Mdot_0 * self.Omega / (2. * pi) * (1 - (Rmin / R)**0.5) + Wrphin * (Rmin / R)**2.)
        return Wphi

    def torque(self, R):
        """Calculate the torque at a given radius by joining the inner and outer SS73 disks solutions
        Parameters
        ----------
        R: float or array-like
            The radius at which to calculate the torque.
        Returns
        -------
        float or array-like
            The calculated torque.
        """
        WrphiRsph = self.innertorque(self.Rsph)
        Wrphi = np.where(R > self.Rsph, self.outerTorque(R, -WrphiRsph, self.Rsph), self.innertorque(R))
        return Wrphi


    def innertorque(self, R):
        """Calculate the torque at a given radius.

        Parameters
        ----------
        R: float or array-like
            The radius at which to calculate the torque.

        Returns
        -------
        float or array-like
            The calculated torque.
        """
        # 2.5 = 5/2
        # 3/2 = 1.5
        Rmin = self.Rmin * self.CO.Risco
        Wrphiinner = self.Mdot * self.Omega / (4. * np.pi) * (1. - (R / Rmin)**2.5) / (1. + 1.5 * (R / Rmin)**2.5)
        return Wrphiinner
    
    
    def Mdot_R(self, R):
        """Calculate the mass-transfer rate at a given radius.

        Parameters
        ----------
        R: float
            The radius at which to calculate the mass-transfer rate.

        Returns
        -------
        float
            The mass-transfer rate at the given radius.
        """
        Rmin = self.Rmin * self.CO.Risco
        Mdot = np.where(R > self.Rsph, self.Mdot_0, 
                        self.Mdot_0 * (self.Rsph / R)**1.5 * (1 + 1.5 * (R / Rmin)**2.5) / (1 + 1.5 * (self.Rsph / Rmin)**2.5) )
        return Mdot


    def solve(self):
        """Solve the disk equations and compute relevant properties."""
        self.Wrphi = self.torque(self.R)
        self.H = self.height(self.Wrphi)
        self.Qrad = self.Q_rad(self.H)
        self.rho = self.density(self.Wrphi, self.H)
        self.vr = self.v_r(self.Mdot_0, self.H, self.rho, self.R)
        self.P = self.pressure(self.H, self.rho)
        self.T = (3 * self.P / A)** (1/4)


class InnerDisk(NonAdvectiveDisk):
    def __init__(self, *args, name="Inner Disk With Outflows", Wrphi_in=0, ewind=1, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.Wrphi_in = Wrphi_in
        self.ewind = ewind

    def torque(self, R):
        """Calculate the torque at a given radius.

        Parameters
        ----------
        R: float or array-like
            The radius at which to calculate the torque.

        Returns
        -------
        float or array-like
            The calculated torque.
        """
        # 2.5 = 5/2
        # 3/2 = 1.5
        Rmin = self.Rmin * self.CO.Risco
        Wrphi = self.Mdot * self.Omega / (4. * np.pi) * (1. - (R / Rmin)**2.5) / (1. + 1.5 * (R / Rmin)**2.5)
        return Wrphi

    
    def Mdot_R(self, R):
        """Calculate the mass-transfer rate at a given radius.

        Parameters
        ----------
        R: float
            The radius at which to calculate the mass-transfer rate.

        Returns
        -------
        float
            The mass-transfer rate at the given radius.
        """
        Rmin = self.Rmin * self.CO.Risco
        Rsph = self.Rmax * self.CO.Risco
        Mdot = self.Mdot_0 * (Rsph / R)**1.5 * (1 + 1.5 * (R / Rmin)**2.5) / (1 + 1.5 * (Rsph / Rmin)**2.5)
        return Mdot


    def solve(self, **kwargs):
        self.Mdot = self.Mdot_R(self.R)
        self.Wrphi = self.torque(self.R)
        self.H = self.height(self.Wrphi)
        self.Qrad = self.Q_rad(self.H)


class InnerDiskODE(NonAdvectiveDisk):
    def __init__(self, *args, name="Inner Disk With Outflows", Wrphi_in=0, ewind=1, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.Wrphi_in = Wrphi_in
        self.ewind = ewind

    def _torque_derivative(self, Mdot, Wrphi, R):
        """Derivative of the torque
        Derived from Equation (4)
        Parameters
        ----------
        Mdot: float,
            Mass-transfer rate at a given radius
        Wrphi: float
            The value of the torque at a given radius
        R: float
            The radius
        w: float
            Keplerian angular velocity at the given radius
        """
        return -(Mdot * self.CO.omega(R) / (4 * pi)  + 2 * Wrphi) / R

    def Mdotprime(self, Wrphi, R):
        """From equation 12 and replacing Qrad using Equation 8 and the Pressure using Equation 9
        The derivative of Mdot is positive (it increases as we move outward), because Wrphi is negative, here a negative sign must be kept (it also follows from the derivation)
        All parameters in cgs units
        Parameters
        ----------
        Wrphi:float or array-like,
            The value of the torque at a given radius
        R: float,
            Radius
        """
        return -6. * np.pi * Wrphi / (self.CO.omega(R) * R)

    def bc(self, ya, yb):

        return np.array([
            yb[0] - self.Mdot_0,      # Mdot(Rsph) = Mdot_0
            ya[1] - self.Wrphi_in,   # W_rphi(Rmin) = Wrphi_in
        ])

    def ode(self, x, y):
        """ODE system for BVP solver with parameter p=[Rsph]"""
        R = x
        Mdot = y[0]
        Wrphi = y[1]
        
        dMdot_dR = self.Mdotprime(Wrphi, R)
        dWrphi_dR = self._torque_derivative(Mdot, Wrphi, R)
        
        # Convert to derivatives wrt x and return as 2D array
        return np.vstack([dMdot_dR, dWrphi_dR])

    def solve(self, **kwargs):
        """Solve the coupled ODEs with BVP solver and integral constraint."""

        # Initial guess for [Mdot, Wrphi]
        Rsph = self.Rmax * self.CO.Risco 
        Rmin = self.Rmin * self.CO.Risco
        M_isco = mass_transfer_inner_radius(self.mdot, self.ewind) * self.Mdot_0
        Mdot_guess = M_isco + (self.Mdot_0 - M_isco)  * (self.R / Rmin) / Rsph 
        #Mdot_guess = self.Mdot_0 * self.R / (self.Rmax * self.CO.Risco)
        Wrphi_guess = -self.Mdot_0 * self.R / (Rsph * Rmin) * self.Omega / (4 * np.pi) * (1 - (self.R / Rmin)**(5/2)) / (1 + 3/2 * (self.R / Rmin)**(5/2))
        Wrphi_guess[0] = self.Wrphi_in

        initial_guess = np.array([Mdot_guess, Wrphi_guess])

        # Solve BVP with parameter [Rsph]
        output = solve_bvp(self.ode, self.bc, self.R, initial_guess, **kwargs)
        
        # Extract solution
        solution = output.sol(self.R)
        self.Mdot = solution[0]
        self.Wrphi = solution[1]
        self.H = self.height(self.Wrphi)
        self.Qrad = self.Q_rad(self.H)



class CompositeDisk(NonAdvectiveDisk):
    """Base class for disks with different inner and outer solutions.
    Extends NonAdvectiveDisk, but requires innerDiskClass and optionally outerDiskClass (default: ShakuraSunyaevDisk).
    """
    def __init__(self, innerDiskClass, outerDiskClass=ShakuraSunyaevDisk, *args, name="CompositeDisk", Wrphi_in=0, ewind=1, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.innerDiskClass = innerDiskClass
        self.outerDiskClass = outerDiskClass
        self.Wrphi_in = Wrphi_in
        self.ewind = ewind
        self.Rsph = None

    
    def adjust_Rsph(self, maxiter=100, reltol=1e-4,):
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

        outerdisk = self.outerDiskClass(self.CO, self.mdot, self.alpha, Rmax=self.Rmax, Rmin=self.Rmin, N=self.N, 
                                       name="Temporary SS Disk", Wrphi_in=0)
        outerdisk.solve()
        L_Ra = outerdisk.L() - self.CO.LEdd

        for i in range(maxiter):
            R_c = (Ra + Rb) / 2.0
            Ninner = (self.R <= R_c * self.CO.Risco).sum()
            # reset the maximum radius
            innerDisk = self.innerDiskClass(self.CO, self.mdot, self.alpha, Rmin=self.Rmin, Rmax=R_c, N=Ninner, 
                                  name="Inner Disk", Wrphi_in=self.Wrphi_in, 
                                  ewind=self.ewind)
            innerDisk.solve()
            Nouter = self.N - Ninner
            if Nouter ==0:
                raise ValueError("Run out of points in Rsph calculation. Increase the number of grid points!")
            # create truncated SS73 with new Wrphi boundary condition
            outerDisk = self.outerDiskClass(self.CO, self.mdot, self.alpha, Rmin=R_c,
                                           Rmax=self.Rmax, N=Nouter, name="Outer Disk", 
                                           Wrphi_in=-innerDisk.Wrphi[-1])
            outerDisk.solve()
            L_Rc = outerDisk.L() - self.CO.LEdd
            err = abs(R_c - Ra)
            if err / R_c < reltol or L_Rc == 0:
                print(f"Solution found after {i+1} iterations with error {err:.2e}")
                return R_c, innerDisk, outerDisk
            if (L_Rc * L_Ra) < 0:
                Rb = R_c
            else:
                Ra = R_c
                L_Ra = L_Rc
        return None
    
    def solve(self):

        Rsph, self.innerDisk, self.outerDisk = self.adjust_Rsph()
        self.Rsph = Rsph * self.CO.Risco
        # Combine solutions
        self.Mdot = np.concatenate((self.innerDisk.Mdot, self.outerDisk.Mdot))
        self.Wrphi = np.concatenate((self.innerDisk.Wrphi, self.outerDisk.Wrphi))
        self.H = self.height(self.Wrphi)
        self.Qrad = self.Q_rad(self.H)
        self.rho = self.density(self.Wrphi, self.H)
        self.vr = self.v_r(self.Mdot, self.H, self.rho, self.R)
        self.P = self.pressure(self.H, self.rho)
        self.T = (3 * self.P / A) ** (1/4)
