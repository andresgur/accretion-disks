import unittest
from accretion_disks.diskwithoutflows import CompositeDisk, InnerDiskODE, InnerDisk
from accretion_disks.shakurasunyaevdisk import ShakuraSunyaevDisk
from accretion_disks.compact_object import CompactObject
import numpy as np
import matplotlib.pyplot as plt

class TestDiskWithOutflows(unittest.TestCase):
    def testManualRsph(self):
        blackhole = CompactObject(M=10, a=0)
        mdot = 100
        for diskclass in [InnerDiskODE, InnerDisk]:
            disk = diskclass(blackhole, mdot=mdot, alpha=0.1, Rmax=1.62 * mdot, N=10000)
            disk.solve()
            ss73 = ShakuraSunyaevDisk(blackhole, mdot=mdot, alpha=0.1, Rmax=10000, Rmin=1.62 * mdot, Wrphi_in=-disk.Wrphi[-1])
            self.assertAlmostEqual(ss73.L() / blackhole.LEdd, 1, delta=0.1, msg="Modified SS73 does not produce Eddington luminosity!")

    def testODE(self):
        """Test that the disk solving the differential equations yields the same results as the analytical one"""
        blackhole = CompactObject(M=10, a=0)
        for mdot in np.arange(10, 100, 10):
            analyticalDisk = CompositeDisk(InnerDisk, ShakuraSunyaevDisk, blackhole, mdot=mdot, Rmax=1e5)
            odeDisk = CompositeDisk(InnerDiskODE, ShakuraSunyaevDisk, blackhole, mdot=mdot, Rmax=1e5)
            np.testing.assert_almost_equal(odeDisk.H[1:] / analyticalDisk.H[1:], np.ones_like(odeDisk.R[1:]), decimal=1)
            np.testing.assert_almost_equal(odeDisk.Mdot[1:] / analyticalDisk.Mdot[1:], np.ones_like(odeDisk.R[1:]), decimal=1)
            np.testing.assert_almost_equal(odeDisk.Wrphi[1:] / analyticalDisk.Wrphi[1:], np.ones_like(odeDisk.R[1:]), decimal=1)

    def testDerivedRsphL(self):
        blackhole = CompactObject(M=10, a=0)
        mdot = 1000
        for diskclass in [InnerDisk, InnerDiskODE]:
            disk = CompositeDisk(diskclass, ShakuraSunyaevDisk, blackhole, mdot=mdot, N=10000, Rmax=1e5)
            L = disk.L(Rmin = disk.Rsph)
            self.assertAlmostEqual(L / blackhole.LEdd, 1, delta=0.1, msg="L(R > Rsph) does not mach the Eddington luminosity!")
            L = disk.outerDisk.L()
            self.assertAlmostEqual(L / blackhole.LEdd, 1, delta=0.1, msg="The outer disk luminosity does not mach the Eddington luminosity!")
            self.assertAlmostEqual(disk.Rsph / blackhole.Risco / mdot, 1.62, delta=0.1)
    

    def testEnergyConserved(self):
        """Test that the energy generation from the torque matches the radiative output Qrad"""
        mdot =10
        blackhole = CompactObject(M=10, a=0)
        for diskclass in [InnerDisk, InnerDiskODE]:
            disk = CompositeDisk(diskclass, ShakuraSunyaevDisk, blackhole, mdot=mdot, Rmax=1e5, N=10000)
            Q = - 3./4. * disk.Omega * disk.Wrphi
            np.testing.assert_allclose(disk.Qrad / disk.Qrad, Q / disk.Qrad, rtol=1e-10)


    def testQrad(self):
        """Test that inside Rsph the Qrad from the torque matches the expected analytical expression"""
        blackhole = CompactObject(M=10, a=0)
        mdot = 10
        for diskclass in [InnerDisk, InnerDiskODE]:
            disk = CompositeDisk(diskclass, ShakuraSunyaevDisk, blackhole, mdot=mdot, Rmax=1e4, N=500000)
            dR = disk.R[1] - disk.R[0]
            dMdR = np.gradient(disk.Mdot, dR)
            Qrad = disk.Omega**2 * disk.R / (8 * np.pi) * dMdR
            rrange = disk.R < disk.Rsph
            # avoid the edges where the numerical derivative is not good
            np.testing.assert_allclose(Qrad[rrange][1:-1] / disk.Qrad[rrange][1:-1], np.ones_like(Qrad[rrange])[1:-1],
                                    rtol=0.1, err_msg="Failure in Qrad calculation!")

if __name__ == '__main__':
    unittest.main()