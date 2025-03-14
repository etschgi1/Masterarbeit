from scf_guess_tools import PyEngine, FScore

engine = PyEngine()  # you can switch between engines on-the-fly
molecule = engine.load("ch3.xyz")
final = engine.calculate(molecule, "pcseg-0")

for scheme in engine.guessing_schemes():
    print(scheme)
    initial = engine.guess(molecule, "pcseg-0", scheme)
    f_score = FScore(initial, final)
    print(f_score())

    