from astropy.constants import G, c, M_sun, m_p, sigma_T
import astropy.units as u

Gcgs = G.to(u.cm**3/u.g/u.s**2).value
ccgs = c.to(u.cm/u.s).value
M_suncgs = M_sun.to(u.g).value
m_pcgs = m_p.to(u.g).value
sigma_Tcgs = sigma_T.to(u.cm**2).value
k_T = sigma_Tcgs / m_pcgs  # cm^2/g
A = 7.56 * 10**-15 # erg/cm**3/K**4 radiation constant
