from scf_guess_tools import Backend, load

example_file = "/home/etschgi1/REPOS/Masterarbeit/datasets/QM9/xyz_c5h4n2o2_sorted/dsgdb9nsd_023539.xyz"

mol = load(example_file, backend=Backend.PY)#, basis="6-31G(2df,p)")
print(mol._native.nao)

