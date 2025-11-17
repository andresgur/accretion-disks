from .constants import Gcgs, ccgs, M_suncgs, m_pcgs, sigma_Tcgs
from math import pi
from astropy.units import Quantity
class CompactObject():
    """Base compact object class"""
    
    def __init__(self, M:  float, a: float =0):
        """
        Parameters
        ----------
        M: float or astropy.quantity
            Mass of the compact object in grams
        a: float
            Dimensionless spin of the compact object: 0 for a Scharzschild black hole or 0.998 for a Kerr black hole.
        """
        self.M = M
        self.a = a

    @property
    def M(self):
        return self._M
    
    @property
    def a(self):
        return self._a
    
    @a.setter
    def a(self, value):
        if value > 1:
            raise ValueError(f"Error in setting spin parameter. \nSpin parameter (a={value:.6f}) exceeds maximum allowed value of 1.")
        self._a = value
        self._update_spin()
    

    @M.setter
    def M(self, value):
        if isinstance(value, Quantity):
            self._M = value.to('g').value
        else:
            # by default we assume solar masses
            self._M = float(value) * M_suncgs
        self._update_mass()

    def _update_mass(self):
        self.Rg = self.gravitational_radius()
        self.LEdd = self.eddington_luminosity()
        
    
    def _update_spin(self):
        self.Risco = self.isco_radius() * self.Rg
        eff = self.accretion_efficiency(self.Risco)
        self.Medd = self.LEdd / ccgs**2 / eff
    
    def __str__(self):
        return (f"Compact Object (CO):\n"
            f"Mass: {self.M / M_suncgs:.2e} M_sun\n"
            f"Spin: {self.spin}\n"
            f"Risco: {self.Risco:.2e} cm\n"
            f"Eddington mass-accretion rate: {self.Medd:.2e} erg/s")
    
    def gravitational_radius(self)-> float:
        """Returns the gravitational radius for a given mass in g.
        Parameters
        ----------
        M: float
            Mass of the compact object in grams

        Returns the gravitational radius in cm
        """
        return Gcgs * self.M/ ccgs**2.
    

    def isco_radius(self, )-> float:
        """Returns the ISCO radius for a given mass in units of Rg.
        Returns the radius of the inner most stable orbit in units of Rg
        """
        z1 = 1 + (1 - self.a**2.) ** (1/3) * ((1 + self.a)** (1/3) + (1-self.a) ** (1/3))
        z2 = (3. * self.a ** 2. + z1**2)**0.5
        # this implements the +- sign of a
        return (3. + z2 - self.a * ( (3 - z1) * (3 + z1 + 2 * z2))**0.5 ) 
    

    def eddington_luminosity(self, )-> float:
        """The classical Eddington luminosity for a given mass.
            Parameters
            ----------
            M: float
                Mass in solar units
            Returns the Eddington luminosity in erg/s (cgs)
        """
        return 4 * pi * Gcgs * self.M * m_pcgs * ccgs / sigma_Tcgs
    

    def eddington_accretion_rate(self, R_in: float) -> float:
        """The classical Eddington luminosity for a given mass.

            Parameters
            ----------
            R_in: float or array-like
                The radius at which the accretion flow is cut (either isco or surface of the star) in cm
            Returns
            -------
            Returns the Eddington accretion rate in quantity
        """
        efficiency = self.accretion_efficiency(R_in)
        return self.LEdd / efficiency / ccgs**2.
    

    def accretion_efficiency(self, R: float)-> float:
        """Returns the accretion efficiency. Everything in cgs
        R: float
            Radius of the compact object or innermost stable orbit in cm
        Returns the accretion efficiency
        """
        return Gcgs * self.M  / (2. * ccgs ** 2. * R)
    


    def omega(self, R:float):
        """Calulate the Keplerian angular velocity at a given radius in cgs units

        Parameters
        ----------
        R: float or array-like,
            Radius at which to calculate the velocity in cm

        Returns
        -------
        Returns the Keplerian angular velocity in rad/s
        """
        return (Gcgs* self.M /  R**3) **0.5