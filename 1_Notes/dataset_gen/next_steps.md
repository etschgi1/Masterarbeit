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
- Dichte 
- Overlap speichern. 


ML raten auf der Fock Matrix als first step 
    - Fock Matrix ist prop zu overlaps
    -> Zerlegung der Matrix (Dichte / Fock) - in homo/hetero teile siehe Cartus. Zerlegung für Dichtematrix -> 
    - Fock Matrix enthält auch alle Energien der unbesetzten 
Koeffizientenmatrix als alternative


-> pyscf 6-31g(2df_p) basis verwenden und b3lypg
