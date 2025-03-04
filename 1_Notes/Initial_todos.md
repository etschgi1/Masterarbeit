# Initial Todos: 

## Learning
- [ ] MoMs VO (10/12)
- [ ] Lehtola_2019 lesen
- [ ] Cartus Methodik lesen! 
- [ ] Report Dominik lesen
- [ ] Hückel Theorie ansehen
- [ ] Ralf Meyer cluster - Intro **Ausmachen, wann wir Termin machen auch bzgl. Kostenfunktion** 

DONE
- [x] Get to know Molpro / QChem system on local cluster
- [x] SCF mit Dominik sprechen 

## Doing 
- [ ] Ideen für Kostenfunktion abklären!
- [ ] Arbeitsplatz mit Bettina abklären 
- [x] Example scripts PySCF 
- [ ] Schnittstelle für PySCF Dominik
  - [ ] Ralf um access zum Repo bitten.
  - [ ] Metric in Pyscf implementieren  
- [ ] PyQchem + ASE Packages ansehen
- [ ] Kostenfunktion überlegen, was wir brauchen

DONE
- [X] Buch bestellen  
- [x] .gitignore file for everything else 



- Vortrag -> Histogramme raus -> Aufteilung - Slides: was wollen wir machen - Timeline
  Wie mappen wir unsere Matrix (unterschiedliche Größen) whatever auf unsere 
- Bi-modale Verteilungen sind schwer zu Trainieren weil viele Optimizer davon ausgehen, dass der Fehler konkav ist. 
- Datenset Quatum Machine.org generieren (~ 1000s of molecules)
  - Raussuchen was relevant ist. QM7 (sind aber sehr kleine Molekühle) - vorerst damit mal beginnen. 
  - B3LYP und PBE Functionals sind die am häufigsten genutzen Funktionale?

- Density fitting in PySCF? Wie funktioniert das? Auf den einzelnen Atomen? 


- Cutoff - Basis Set (F12) - wieder zurückgefittet auf Atom-Basen RIFIT und JKFIT 
- First step Transformation von JKFIT integralen (Basis-sets) zu kontrahierten Versionen davon. def2-universal-jkfit und dann zurück projezieren auf Basissets

Action items: 
- Psi4 Ralf Skript - siehe Dichte Matrizen von Notebooks 
- Ziel ist es eine Dichte Matrix aus den Notebooks versuchen in die Atombasis projezieren (hin + her transformieren) - Mittlerer Fehler an den einzelnen Atomen dann und ändert sich dieser Fehler, wenn Basissatz verändert wird (def2-universal-jkfit) 
-  
