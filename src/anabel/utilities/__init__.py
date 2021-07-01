def take_keywords(kwds:dict, *args):
    return {arg: kwds.pop(arg) for arg in args if arg in kwds}

