import unittest
from accretion_disks.shakurasunyaevdisk import ShakuraSunyaevDisk
from accretion_disks.compact_object import CompactObject
import numpy as np
import matplotlib.pyplot as plt

class TestShakuraSunyaevDisk(unittest.TestCase):

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

if __name__ == '__main__':
    unittest.main()