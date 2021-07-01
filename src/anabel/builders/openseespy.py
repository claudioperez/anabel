import sys
import importlib

openseespy_spec = importlib.util.find_spec("openseespylinux.opensees")

class OpenSeesModel:
    def __init__(self,ndf,ndm):
        self._openseespy = importlib.util.module_from_spec(openseespy_spec)
        openseespy_spec.loader.exec_module(self._openseespy)
        if "openseespy.opensees" in sys.modules:
            del sys.modules["openseespy.opensees"]
        if "openseespy" in sys.modules:
            del sys.modules["openseespy"]
        #import openseespy.opensees
        #self._openseespy = openseespy.opensees
        self.__dict__.update({
            k:v for k,v in self._openseespy.__dict__.items() if k[:2] != "__"
        })
        self.model('basic', '-ndm', ndm, '-ndf', ndf)
        #del sys.modules["openseespy"]


