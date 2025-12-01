import unittest
from accretion_disks.advectivedisk import ConservativeInnerDisk
from accretion_disks.shakurasunyaevdisk import ShakuraSunyaevDisk
from accretion_disks.compact_object import CompactObject
import numpy as np
from accretion_disks.constants import ccgs, k_T
import matplotlib.pyplot as plt


class TestAdvectiveDisk(unittest.TestCase):

    def setUp(self):
        self.blackhole = CompactObject(M=10, a=0)

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

    def testEnergyConserved(self):

        Wrphi_in = 0
        Hin = (
            -3 / 4 * Wrphi_in * k_T / ccgs / self.blackhole.omega(self.blackhole.Risco)
        )
        # Hin = 500
        Hin *= 0
        print(Hin / self.blackhole.Risco)
        disk = ConservativeInnerDisk(
            self.blackhole,
            mdot=0.8,
            Rmin=1.1,
            alpha=0.1,
            H_in=Hin,
            N=50000,
            Rmax=1e5,
            Wrphi_in=Wrphi_in,
        )
        print(disk)
        disk.plot()
        plt.show()
        np.testing.assert_allclose(
            np.ones(disk.N),
            disk.Qvis / (disk.Qrad + disk.Qadv),
            rtol=1e-4,
            err_msg="Energy is not conserved!",
        )

    def testTorqueDerivative(self):
        disk = ConservativeInnerDisk(
            self.blackhole, mdot=0.1, alpha=0.1, H_in=1e-5, N=10000
        )
        dWrphi = np.gradient(disk.Wrphi, disk.R)
        np.testing.assert_allclose(
            disk.torque_derivative(disk.R) / dWrphi, np.ones(disk.N), rtol=1e-2
        )

    def testHeightDerivative(self):
        disk = ConservativeInnerDisk(
            self.blackhole, mdot=0.1, alpha=0.1, H_in=0.1, N=20000
        )
        diskss73 = ShakuraSunyaevDisk(self.blackhole, mdot=0.1, alpha=0.1)

        disk.Q_adv()
        H = diskss73.H
        Mdot = diskss73.Mdot
        R = diskss73.R
        Wrphi = diskss73.Wrphi
        dH = (
            1
            / 9.0
            * (
                -3
                * np.pi
                * diskss73.R
                * diskss73.Wrphi
                / (diskss73.Omega * H * diskss73.Mdot)
                - 4 * diskss73.R * np.pi * ccgs / (diskss73.Mdot * k_T)
            )
        )
        # dH = 10./9. * H /diskss73.R - np.pi * diskss73.R * diskss73.Wrphi / (3. * diskss73.Omega * H * diskss73.Mdot) - diskss73.Mdot * diskss73.Omega * H / (36. * np.pi * diskss73.R * diskss73.Wrphi) - 4 * np.pi * R *ccgs / (9. * Mdot *k_T)
        # Hprimecalc = np.gradient(H, diskss73.R)
        # np.testing.assert_allclose((dH / Hprimecalc)[10:-10], np.ones(diskss73.N)[10:-10], rtol=1e-1)
        np.testing.assert_allclose(
            (dH)[10:-10], np.zeros(diskss73.N)[10:-10], rtol=1e-1
        )


if __name__ == "__main__":
    unittest.main()
