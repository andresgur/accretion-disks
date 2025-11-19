import unittest
from accretion_disks.shakurasunyaevdisk import ShakuraSunyaevDisk
from accretion_disks.compact_object import CompactObject
import numpy as np
import matplotlib.pyplot as plt

class TestShakuraSunyaevDisk(unittest.TestCase):



    def scale_height(self, disk, Mr):
        """Equation 18 from Lipunova+99, works for both sub and super critical disks as long as advection is neglected
            Just replace Mdot(R) by the appropiate calculation (i.e. without or with outflows). Equivalent to Eq 7 from Pountanen+2007
            Everything in cgs units
        Parameters
        ----------
        m_r:float,
            (Dimensionless) Mass-transfer rate at every radii
        R: float,
            Radii at which the scale height is to be calculated
        M: float
            Mass of the compact object
        """
        R0 = disk.CO.Risco
        Rs = 2 * disk.CO.Rg
        efficiency = disk.CO.accretion_efficiency(R0)
        m_r = Mr / disk.CO.MEdd
        H = Rs * m_r * 3 / 4 / efficiency * (1 - np.sqrt(R0/ disk.R))
        return H

    def testMaxQ2(self):
        blackhole = CompactObject(M=10, a=0)
        disk = ShakuraSunyaevDisk(blackhole, mdot=0.1, alpha=0.1)
        disk.solve()
        max_Qr2 = np.argmax(disk.Qrad * disk.R**2)
        self.assertAlmostEqual(disk.R[max_Qr2] / blackhole.Risco, 2.25, delta=0.1)

    def testEnergyConserved(self):
        blackhole = CompactObject(M=10, a=0)
        disk = ShakuraSunyaevDisk(blackhole, mdot=0.1, alpha=0.1)
        disk.solve()
        Q = - 3./4. * disk.Omega * disk.Wrphi
        np.testing.assert_allclose(disk.Qrad / disk.Qrad, Q / disk.Qrad, rtol=1e-5)


    def testLuminosity(self):
        blackhole = CompactObject(M=10, a=0)
        for mdot in np.arange(0.1, 0.9, 0.1):
            disk = ShakuraSunyaevDisk(blackhole, mdot=mdot, alpha=0.1)
            disk.solve()
            L = 4 * np.pi * np.sum(disk.Qrad * disk.R) * (disk.R[1] - disk.R[0])
            self.assertAlmostEqual(L / blackhole.LEdd, mdot, delta=0.05)


    def testScaleHeight(self):
        blackhole = CompactObject(M=10, a=0)
        for mdot in np.arange(0.1, 0.9, 0.5):
            disk = ShakuraSunyaevDisk(blackhole, mdot=mdot, alpha=0.1, N=50000, Rmax=100000)
            disk.solve()
            H = self.scale_height(disk, disk.Mdot_0)
            np.testing.assert_allclose(disk.H[1:] / H[1:], np.ones_like(disk.H[1:]), rtol=1e-2, atol=1e-2)


if __name__ == '__main__':
    unittest.main()