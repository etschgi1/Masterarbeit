# Fragen
-> Um von Fock Matrix zur density zu kommen muss man ja einmal FC = SCe lösen -> sprich eine Iteration des Problems


# Using Fourier NN 
https://arxiv.org/pdf/2010.08895

💡 What can you do?
✅ Strategy 1: Fix the basis set
Train and deploy within one basis set (e.g., STO-3G or 6-31G).

This keeps S → F mapping stable.

You can still learn across many molecules.

✅ Strategy 2: Normalize across basis sets
Map S and F into a basis-invariant representation (e.g., density projected onto real-space grid).

Use that for FNO input/output.

This lets you generalize across molecules and basis sets.

✅ Strategy 3: Use graph/Fourier hybrid models
If basis set changes are essential, consider combining:

Graph Neural Networks (for basis structure)

Fourier components (for function representation)

# Try transfer Learning!
-> Feature Extraction 
-> (or worse - bc more tuning) Fine-tuning
For fine tuning you usually take the last few layers and retrain them because they have the coarse details (while the first and mid only have fine details ...)

# Data augmentation -> flip around the atoms in the xyz file to get more data!
