def singleton(cls):
    instances = {}
    def wrapper(*args, **kwargs):
        key = (args, frozenset(kwargs.items()))
        if key not in instances:
            instances[key] = cls(*args, **kwargs)
        return instances[key]
    return wrapper
