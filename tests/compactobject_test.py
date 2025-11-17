import unittest
from accretion_disks.constants import M_suncgs
from accretion_disks.compact_object import CompactObject

class TestCompactObject(unittest.TestCase):
    def test_Mass_setter(self):
        M = 10
        co = CompactObject(M=M, a=0.0)
        self.assertEqual(co.M, M * M_suncgs)

    def test_isco_schwarzchild(self):
        co = CompactObject(M=10, a=0)
        Risco_expected = 6  # in units of Rg for a=0
        self.assertAlmostEqual(co.Risco / co.Rg, Risco_expected)

    def test_isco_kerr(self):
        co = CompactObject(M=10, a=1.)
        Risco_expected = 1  # in units of Rg for a=0
        self.assertAlmostEqual(co.Risco / co.Rg, Risco_expected)

    def test_eddington_luminosity(self):
        M = 10
        co = CompactObject(M=M, a=0)
        L_edd_expected = 1.26 * M  # in 10**38 erg/s (wikipedia)
        self.assertAlmostEqual(co.LEdd / 10**38, L_edd_expected, delta=0.1)


if __name__ == '__main__':
    unittest.main()