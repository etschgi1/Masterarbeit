# Dataset gen 
Dipole moment / Quadrupole moment etc. vergleichen???

## Neuer Basissatz 
Geometrie wird gleich verwendet aber wir verwenden für das machine Learning ein anderes Basisset (nur sphärisch). 

- auschecken ob sich die Geometrie ändert für die Moleküle -> für ~10 Moleküle checken ob sich was an der Geometrie ändert. 
Wir nehmen: WB97X_V
und Basisset: aug-cc-pVDZ

Geometrieoptimierung - 
rmsd - packet verwenden um den rms zwischen dne beiden molekühlen anordnen
!!!!xyz file müssen die gleiche Reihenfolge der Atome haben


Schrittweite kleiner einstellen. 

- Fock Matrix implementieren


TEST mit WB97X_V & aug-cc-pVDZ: 
Got 6095 files from ../datasets/QM9/xyz_c7h10o2!
Start pyscf
Loaded mol from /tmp/dsgdb9nsd_082759_valid.xyz (0 / 6095)
converged SCF energy = -423.086570214557
Diff to reference energy: -0.06483778544287588
SCF Energy: -423.0865702145571
Reference Energy: -423.151408
Loaded mol from /tmp/dsgdb9nsd_040155_valid.xyz (1 / 6095)
converged SCF energy = -423.052071219599
Diff to reference energy: -0.0802677804014138
SCF Energy: -423.0520712195986
Reference Energy: -423.132339
Diff to reference energy: -0.0802677804014138
SCF Energy: -423.0520712195986
Reference Energy: -423.132339
Time elapsed (pyscf): 247.97326636314392

