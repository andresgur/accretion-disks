from .basedisk import NonAdvectiveDisk
from .shakurasunyaevdisk import ShakuraSunyaevDisk
from math import pi
from .constants import k_T, A, ccgs, Gcgs
from scipy.integrate import solve_bvp
import numpy as np
import warnings



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


class DiskWithOutflows(NonAdvectiveDisk):
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
    


    def torque(self, R):
        """Calculate the torque at a given radius.

        Parameters
        ----------
        R: float
            The radius at which to calculate the torque.

        Returns
        -------
        float
            The calculated torque.
        """
        # 2.5 = 5/2
        # 3/2 = 1.5
        Rmin = self.Rmin * self.CO.Risco
        return self.Mdot * self.Omega / (4. * np.pi) * (1. - (R / Rmin)**2.5) / (1. + 1.5 * (R / Rmin)**2.5)
    
    def _gR(self, R):
        Rmin = self.Rmin * self.CO.Risco
        r = R / Rmin
        rsph = self.Rsph / Rmin
        gr = self.mdot /3 * r**(3/2) / rsph * (1 - r**(-5/2))/ (1 + 2/3 * rsph**(-5/2))
        return gr

    def torquePoutanen(self, R):
        """Analytical expression for the torque when mass loss is included.

        Parameters
        ----------
        R: float
            The radius at which to calculate the torque.

        Returns
        -------
        float
            The calculated torque based on Poutanen+2007.
        """
        
        #
        Rmin = self.Rmin * self.CO.Risco
        g0 = (Gcgs * self.CO.M * Rmin)**0.5 * self.CO.MEdd
        grsph = self._gR(self.Rsph)
        r = R / Rmin
        gr = np.where(self.R > self.Rsph, grsph / g0 + self.mdot  * (r**0.5 - (self.Rsph / self.R)**0.5), self._gR(self.R)) * g0
        Wrphi = gr / (2 * np.pi * self.R**2)
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



class DiskWithOutflowsODE(NonAdvectiveDisk):
    def __init__(self, *args, name="Disk With Outflows", Wrphi_in=0, ewind=1, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.Wrphi_in = Wrphi_in
        self.ewind = ewind
        # Approximative solution for Rpsh based on Poutanen+2007
        deltaR = np.diff(self.R)[0]
        disk = ShakuraSunyaevDisk(*args, name="Temporary SS Disk", **kwargs)
        disk.solve()
        Qrad = disk.Qrad
        Ltot = 4 * np.pi * np.sum(Qrad * self.R) * deltaR
        if Ltot < self.CO.LEdd:
            warnings.warn("Spherization radius is not defined as the disk does not exceed its Eddington limit. Either you need to increase the outer radius, or the mass-transfer rate")
            # set disk to Shakura-Sunyaev
            self.Rsph = self.Rmax * self.CO.Risco
        else:
            self.Rsph = 1.62 * self.mdot * self.CO.Risco
        
        # For BVP solver with integral constraint
        self._current_x = None
        self._current_y = None
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
        print("Full luminosity", fourpideltaR * QradR[R_range].sum() / self.CO.LEdd)

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
        M_isco = mass_transfer_inner_radius(self.mdot, self.ewind) * self.Mdot_0
        Mdot_guess = M_isco + (self.Mdot_0 - M_isco)  * (self.R / self.Rmin) / self.Rsph
        Rmin = self.Rmin * self.CO.Risco
        Wrphi_guess = -self.Mdot_0 * self.R / (self.Rsph * Rmin) * self.Omega / (4 * np.pi) * (1 - (self.R / Rmin)**(5/2)) / (1 + 3/2 * (self.R / Rmin)**(5/2))
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
        self.rho = self.density(self.Wrphi, self.H)
        self.v_r = self.v_r(self.Mdot, self.H, self.rho, self.R)
        self.P = self.pressure(self.H, self.rho)
        self.T = (3 * self.P / A) ** (1/4)