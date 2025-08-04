class DB(object):
    r"""Abstract base class to create and manipulate database
    """
    def __init__(self, filename, mode = 'a', **kwargs):
        pass

    @property
    def group(self):
        return self._group

    @group.setter
    def group(self, value):
        self._group = value

    @property
    def names(self):
        dicts = {}
        for k, v in self.fh.items():
            dicts[k] = list(v.keys())
        names = [[k, k1] for k in dicts for k1 in dicts[k]]
        return names

    def create_group(self, name, fh = None):
        fh = fh or self.fh
        if isinstance(name, str):
            name = name.split('/')
        if len(name) == 1 :
            if name[0] in fh :
                return fh[name[0]]
            else :
                return fh.create_group(name[0])
        else :
            return self.create_group(name[1:], self.create_group(name[0], fh))

    def get_group(self, name, fh = None):
        fh = fh or self.fh
        if isinstance(name, str):
            name = name.split('/')
        if len(name) == 1 :
            if name in fh :
                return fh[name[0]]
            else :
                raise AttributeError(f"{name} not in the file.")
        else :
            return self.get_group(name[1:], self.get_group(name[0], fh))
