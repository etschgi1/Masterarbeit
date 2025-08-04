import argparse
commands = {
        '--traj2db' : 'qmlearn.cui.traj2db',
        }

def get_args():
    parser = argparse.ArgumentParser(description='QMLearn')
    #-----------------------------------------------------------------------
    parser.description = 'QMLearn tasks :\n' + '\n\t'.join(commands.keys())
    parser.add_argument('--traj2db', dest='traj2db', action='store_true', default=False,
            help='Create the QMLearn database from trajectory file')
    #-----------------------------------------------------------------------
    args = parser.parse_args()
    return args


def main():
    import sys
    from importlib import import_module
    for job in commands :
        if job in sys.argv :
            module = import_module(commands[job])
            command = getattr(module, 'main')
            return command()
    else :
        get_args()
