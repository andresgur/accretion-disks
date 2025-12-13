import unittest
from accretion_disks.shakurasunyaevdisk import ShakuraSunyaevDisk
from accretion_disks.compact_object import CompactObject
from accretion_disks.constants import Gcgs, ccgs, k_T
import numpy as np


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
        H = 0.75 * Rs * m_r / efficiency * (1 - (R0 / disk.R) ** 0.5)
        return H

    def testMaxQ2(self):
        blackhole = CompactObject(M=10, a=0)
        disk = ShakuraSunyaevDisk(blackhole, mdot=0.1, alpha=0.1, Rmax=1e4, N=1000000)
        max_Qr2 = np.argmax(disk.Qrad * disk.R**2)
        self.assertAlmostEqual(disk.R[max_Qr2] / blackhole.Risco, 2.25, delta=0.1)

    def testEnergyConserved(self):
        blackhole = CompactObject(M=10, a=0)
        disk = ShakuraSunyaevDisk(blackhole, mdot=0.1, alpha=0.1)
        Q = -3.0 / 4.0 * disk.Omega * disk.Wrphi
        np.testing.assert_allclose(disk.Qrad / disk.Qrad, Q / disk.Qrad, rtol=1e-5)

    def testLuminosityPropTomdot(self):
        blackhole = CompactObject(M=10, a=0)
        for mdot in np.arange(0.1, 0.9, 0.1):
            disk = ShakuraSunyaevDisk(blackhole, mdot=mdot, alpha=0.1, N=50000)
            L = disk.L()
            self.assertAlmostEqual(
                L / blackhole.LEdd,
                mdot,
                delta=0.05,
                msg="Error luminosity is not proportional to mdot!",
            )

    def testScaleHeight(self):
        blackhole = CompactObject(M=10, a=0)
        for mdot in np.arange(0.1, 0.9, 0.5):
            disk = ShakuraSunyaevDisk(blackhole, mdot=mdot, alpha=0.1, Rmax=100000)
            H = self.scale_height(disk, disk.Mdot_0)
            np.testing.assert_allclose(
                disk.H[1:] / H[1:], np.ones_like(disk.H[1:]), rtol=1e-2, atol=1e-2
            )

    def testTorqueDerivative(self):
        blackhole = CompactObject(M=10, a=0)
        disk = ShakuraSunyaevDisk(blackhole, mdot=0.5, alpha=0.1, Rmax=100000)

        np.testing.assert_allclose(
            disk.torque_derivative(disk.R) / np.gradient(disk.Wrphi, disk.R),
            np.ones(disk.N),
            rtol=1e-2,
            err_msg="Derivative of the torque is not right!",
        )

    def testmdotsetter(self):

        blackhole = CompactObject(M=10, a=0)
        mdot = 0.5
        disk = ShakuraSunyaevDisk(blackhole, mdot=mdot, alpha=0.1, N=200000)
        L = disk.L()
        self.assertAlmostEqual(
            L / blackhole.LEdd,
            mdot,
            delta=0.05,
            msg="Error luminosity is not proportional to mdot!",
        )
        newmdot = 0.2
        disk.mdot = newmdot
        L = disk.L()
        self.assertAlmostEqual(
            L / blackhole.LEdd,
            newmdot,
            delta=0.05,
            msg="Error luminosity is not proportional to mdot!",
        )

    def testPressure(self):
        blackhole = CompactObject(M=10, a=0)
        mdot = 0.5
        disk = ShakuraSunyaevDisk(blackhole, mdot=mdot, alpha=0.1, N=200000)
        P = -disk.Wrphi / 2 / disk.H / disk.alpha
        np.testing.assert_allclose(P / disk.P, np.ones(disk.N), rtol=1e-2, atol=1e-3)

    def testAlphaRaises(self):
        blackhole = CompactObject(M=10, a=0)
        with self.assertRaises(ValueError):
            # negative alpha
            ShakuraSunyaevDisk(
                blackhole,
                mdot=10,
                alpha=-1,
            )
            # rmin > rmax
            ShakuraSunyaevDisk(blackhole, mdot=0.1, alpha=0.1, Rmin=10, Rmax=5)
            # rmin < 1
            ShakuraSunyaevDisk(blackhole, mdot=0.1, alpha=0.1, Rmin=0.1, Rmax=5)
            # Wrphi >0
            ShakuraSunyaevDisk(
                blackhole, mdot=0.1, alpha=0.1, Rmin=0.1, Rmax=5, Wrphi_in=1
            )

    def testalphasetter(self):
        blackhole = CompactObject(M=10, a=0)
        mdot = 0.5
        disk = ShakuraSunyaevDisk(blackhole, mdot=mdot, alpha=0.1, N=200000)
        rho = disk.rho
        P = disk.P
        T = disk.T
        disk.alpha = 0.2
        self.assertTrue(np.all(rho > disk.rho))
        self.assertTrue(np.all(P > disk.P))
        self.assertTrue(np.all(T > disk.T))

    def test_hydrostatic_eq(self):

        blackhole = CompactObject(M=10, a=0)
        mdot = 0.5
        disk = ShakuraSunyaevDisk(blackhole, mdot=mdot, alpha=0.1, N=200000)
        disk.solve()

        leftside = disk.P / (disk.rho * disk.H)
        rightside = Gcgs * blackhole.M * disk.H / disk.R**3

        np.testing.assert_allclose(
            leftside, rightside, err_msg="Disk is not in hydrostatic equilibrium!"
        )

    def test_mass_flux(self):

        blackhole = CompactObject(M=10, a=0)
        mdot = 0.5
        disk = ShakuraSunyaevDisk(blackhole, mdot=mdot, alpha=0.1, N=200000)
        disk.solve()

        leftside = disk.Mdot
        rightside = 4 * np.pi * disk.R * disk.H * disk.rho * disk.vr
        np.testing.assert_allclose(
            leftside, rightside, err_msg="Disk does not conserve mass flux!"
        )

    def test_Qrad(self):
        blackhole = CompactObject(M=10, a=0)
        mdot = 0.5
        disk = ShakuraSunyaevDisk(blackhole, mdot=mdot, alpha=0.1, N=200000)
        disk.solve()

        leftside = disk.Qrad
        rightside = disk.P * ccgs / k_T / disk.rho / disk.H
        np.testing.assert_allclose(
            leftside, rightside, err_msg="Disk Qrad is not preserved!"
        )


if __name__ == "__main__":
    unittest.main()
