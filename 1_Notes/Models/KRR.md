# Notes on training for KRR
The models are not really optimized to be used in the MultiOutputRegressor for large systems in sklearn. For 49x49 this is still possible in okish time. 

Fitting KRR with rbf / poly 2 and poly 3 kernel. 
We obtain low -> very low alpha / gamma values which our method 