ich habe soeben einen Zugang für Dich am alten Mini-Cluster angelegt,
auf dem die Programme "Molpro" und "Q-Chem" installiert sind, zum
Kennenlernen einer Computer-Chemie-Software.

Aufruf von Q-Chem oder Molpro z.B. mit 2 Threads über

qchem -nt 2 inputfile > outputfile &

molpo -n 2 inputfile

Hier bitte inputfile und outputfile mit passenden Dateinamen ersetzen;
das & am Schluss erlaubt eine Ausführung im Hintergrund, auch wenn Du
Dich ausloggst. Bitte nicht mehr als 4 Threads verwenden, vorher mit dem
Befehl "top" checken, ob eh nix los ist.

Einstieg über

ssh ewachmann@129.27.156.7 von einem Stadard-Terminal-Programm. Auf
Windows muss man da vorher eine SSH-Shell installieren.

PW ist Vorname, bitte anschließend ändern mittels "passwd".

Infos zu den Paketen:

https://www.molpro.net/manual/doku.php?id=table_of_contents

https://manual.q-chem.com/5.4/


# Info Session 
- [ ] z-Matrix Format

Jobtype: SP -> single point
OPT -> optimization (for geometry)

Basis sets: https://www.basissetexchange.org/
All Electron / Orbital with ECP (innere Schalen werden durch ein effektives Potential ersetzt)

*Ase* Python -> Package - Schnittstellen zu verschiedenen Programmen (z.b. QChem, etc) -> erstellt direkt die Input files + liest Output files aus. 

*PyQchem* -> ähnlich aber wesentlich ausführlicher weil programmspezifisch. 
