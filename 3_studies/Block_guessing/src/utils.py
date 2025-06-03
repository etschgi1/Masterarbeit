VERBOSE_LEVEL = 2

def dprint(printlevel, *args, **kwargs):
    """Customized printing levels"""
    global VERBOSE_LEVEL
    if printlevel <= VERBOSE_LEVEL:
        print(*args, **kwargs)

def set_verbose(level: int):
    global VERBOSE_LEVEL
    VERBOSE_LEVEL = level