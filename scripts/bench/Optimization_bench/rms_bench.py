import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
optimized_path = "./optimized"
initial_path = "./test_data"

def load_files(p1, p2): 
    """
    Load the files from the given paths.
    """
    f1, f2 = [], []
    for file in os.listdir(p1):
        if file.endswith(".xyz"):
            f1.append(os.path.join(p1, file))
    for file in os.listdir(p2):
        if file.endswith(".xyz"):
            f2.append(os.path.join(p2, file))
    f_both = set(os.path.basename(f) for f in f1).intersection(set(os.path.basename(f) for f in f2))
    out = []
    for f in f_both: 
        out.append((os.path.join(p1, f), os.path.join(p2, f)))
    return out, f_both

def get_rms(init, opt): 
    """
    Get the RMS difference between the initial and optimized files.
    """
    import subprocess
    files, names = load_files(init, opt)
    rms = {}
    for (f1, f2), name in zip(files, names):
        try:
            result = subprocess.run(
                ["calculate_rmsd", f1, f2],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            ).stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"calculate_rmsd failed for {f1} {f2}")
            print("Return code:", e.returncode)
            print("Output:", e.stdout)
            raise
        print(f"Got output raw: {result}")
        try:
            rmsd = float(result)
            rms[name] = rmsd
        except ValueError:
            raise ValueError(f"Error parsing RMSD value for {f1} {f2}: {result}")
    return rms
    
    
    
if __name__ == "__main__": 
    rms = get_rms(initial_path, optimized_path)
    print(rms)