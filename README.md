# Accretion disk solver
This code allows easy access to the disk Equations proposed by Lipunova+99 (which include the seminal thin-disk solution of Shakura & Sunyaev+1973). All disks are assumed to be optically thick and radiation-pressure suported.
The code solves the differential Equations for four types of disk:
- Conservative and non-advective disk (the analytical Shakura & Sunyaev 73 thin disk solution)
- Non-conservative and non-advective disk ("Slim" disk with Outflows, although it is solved numerically, an analytical solution exists as shown in Lipunova+99)
- Conservative and advective disk ("Slim" disk without outflows, no analytical solution exists, numerically solved)
- Non-conservative and advective disk ("Slim" advective disk with outflows, no analytical solution exists, numerically solved)

The code can be used to retrieve quantities of interest for observers such as the height, density, radial velocity, radiative output, etc.

The pdf in the docs folder shows the derivation of the final Equations that go into the solver.

## Installation

### Git Clone
To clone the repository, run the following command:
```bash
git clone https://github.com/andresgur/accretion-disks .
```

### Pip Install
Navigate to the cloned directory and install the package using pip:
```bash
cd accretion-disks
pip install .
```

## Usage
See the notebook in docs/examples.ipynb

```python
from accretion_disks.shakurasunyaevdisk import ShakuraSunyaevDisk
from accretion_disks.diskwithoutflows import CompositeDisk
from accretion_disks.compact_object import CompactObject
import numpy as np

# Define a 10 solar mass black hole with a spin parameter a=0.5
co = CompactObject(10, a=0.5)


## Standard shakura and Sunyeave disk (only Zona A i.e. radiation pressure dominated)
disk_SS73 = ShakuraSunyaevDisk(CO=co, mdot=0.1, alpha=0.1, N=10000)

# Plot the disk properties (e.g., H/R, Qrad, etc.)
fig, axes = disk_SS73.plot()

# Retrieve the total disk luminosity
print(f"Total disk luminosity {disk_SS73.L() / co.LEdd:.2f} LEdd")
# Output: Total disk luminosity 0.50 LEdd

## Disk with Outflows
superEdd_disk = CompositeDisk(CO=co, mdot=1000, alpha=0.1, N=10000)

# Plot the disk properties
fig, axes = superEdd_disk.plot()

# Retrieve properties like the maximum energy release radius and total luminosity
maxQ = np.argmax(superEdd_disk.Qrad * superEdd_disk.R**2)
print("Maximum energy released occurs at", f"{superEdd_disk.R[maxQ] / superEdd_disk.CO.Risco:.2f}", f"with H/R = {superEdd_disk.H[maxQ]/superEdd_disk.R[maxQ]:.2f}")
# Output: Maximum energy released occurs at 2.25 with H/R = 0.22

print(f"Total disk luminosity {superEdd_disk.L() / co.LEdd:.2f} LEdd")
# Output: Total disk luminosity 0.50 LEdd