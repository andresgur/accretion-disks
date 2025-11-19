import unittest
from accretion_disks.diskwithoutflows import DiskWithOutflows, DiskWithOutflowsODE
from accretion_disks.shakurasunyaevdisk import ShakuraSunyaevDisk
from accretion_disks.compact_object import CompactObject
from accretion_disks.constants import Gcgs
import numpy as np
import matplotlib.pyplot as plt

class TestDiskWithOutflows(unittest.TestCase):


    def testRsph(self):
        blackhole = CompactObject(M=10, a=0)
        mdot = 1000
        disk = DiskWithOutflows(blackhole, mdot=mdot, alpha=0.1, N=50000, Rmax=100000)
        disk.solve()

        Rsph = disk.get_spherization_radius(disk.Qrad)

        L = disk.L(Rsph)
        self.assertAlmostEqual(L / blackhole.LEdd, 1, delta=0.1)
        self.assertAlmostEqual(Rsph / blackhole.Risco / mdot, 1.62, delta=0.1)
    

    def testEnergyConserved(self):
        """Test that the energy generation from the torque matches the radiative output Qrad"""
        mdot =10
        blackhole = CompactObject(M=10, a=0)
        disk = DiskWithOutflows(blackhole, mdot=mdot, alpha=0.1)
        disk.solve()
        Q = - 3./4. * disk.Omega * disk.Wrphi
        np.testing.assert_allclose(disk.Qrad / disk.Qrad, Q / disk.Qrad, rtol=1e-10)


    def testQrad(self):
        """Test that inside Rsph the Qrad from the torque matches the expected analytical expression"""
        blackhole = CompactObject(M=10, a=0)
        mdot = 10
        disk = DiskWithOutflows(blackhole, mdot=mdot, alpha=0.1, N=1000000, Rmax=100)
        disk.solve()
        deltaR = disk.R[1] - disk.R[0]
        dMdR = np.gradient(disk.Mdot, deltaR)
        Qrad = disk.Omega**2 * disk.R / (8 * np.pi) * dMdR
        # avoid the edges where the numerical derivative is not good
        np.testing.assert_allclose(np.ones_like(Qrad[disk.R < disk.Rsph])[1:-1], Qrad[disk.R < disk.Rsph][1:-1] / disk.Qrad[disk.R < disk.Rsph][1:-1],
                                   rtol=1e-4)

if __name__ == '__main__':
    unittest.main()