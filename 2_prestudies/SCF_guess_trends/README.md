# SCF guess trends

Was wollten wir mit diesem Projekt erreichen?
War gedacht als vorarbeit um zu sehen ob unterschiedliche guess-Methoden auf unterschiedlichen Molekühlen (organisch / anorganisch / organometallisch) unterschiedlich gut funktionieren. Insights siehe Notebook! 

Summa-Summarum: ist eigentlich nichts großartiges bei rausgekommen. Es gibt einige Unterschiede zwischen den Molekühlen (vor allem im Bezug auf die Atom-Anzahl). Zudem ist spannend, dass die Verteilung der besten Guess-Methoden sich nicht so stark unterscheidet für die beiden unterschiedlichen Methoden HF und revTPSSh.

Bimodale - Verteilungen sind zusätzlich vermutlich auch noch schwerer zu fitten wenn wir in Richtung eines ML-Decision Models bzgl. bestes klassischen guesses gehen! 

*Weitere Insights*: 
- HF und revTPSSh haben eine annähernd gleiche Verteilung der jeweils besten Guess-Methoden
- Verwandte Basissets (pcseg-(0,1,2(aug))) haben ebenfalls eine ähnliche Verteilung der besten Guess-Methoden -> STO-3G nutzt verstärkt sadno und kaum hückel (für besten f-score guess)
- Es ergeben sich bessere guess-methoden abhängig von der größe der Moleküle
- Limitierte Einblicke im Bezug auf funktionale Gruppen (zu kleiner Datenset um statistisch wirklich aussagekräftige Ergebnisse zu erhalten)