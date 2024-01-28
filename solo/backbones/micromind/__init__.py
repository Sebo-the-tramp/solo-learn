from .micromind import phinet as default_phinet
from .micromind import xinet as default_xinet

def phinet(method, *args, **kwargs):
    return default_phinet(*args, **kwargs)

def xinet(method, *args, **kwargs):
    return default_xinet(*args, **kwargs)

__all__ = ["phinet", "xinet"]
