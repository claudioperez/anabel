import inspect
import functools
from pathlib import Path

def _test_source(obj):
    try:
        return ("anabel" in inspect.getmodule(obj).__name__) and (not inspect.ismodule(obj))
    except:
        return False

class CommandLineDriver:
    def __init__(self, *args):
        self._name = "<driver file>"
        self._objects = {}
        self._procedures = {}
        self._definitions = {}
        self.appendages = []

    def _usage(self):
        """Print usage message and exit"""
        print("""usage: twin [OPTIONS]""")
    
    def procedure(self, op):
        @functools.wraps(op)
        def _wrapper(*args, **kwds):
            kwds.update({
                k: self._objects[k]
                for k in inspect.getargspec(op)[0]
                    if k in self._objects
            })
            return op(*args, **kwds)
        self._procedures[op.__name__] = _wrapper
        return op

    def include(self, *modules):
        for module in modules:
            self._objects.update(inspect.getmembers(module, _test_source))

    def let(self, *args, **kwds):
        for arg in args:
            print(vars(arg))
        
        self._objects.update(kwds)

    def run(self):
        import sys
        arg_iter = iter(sys.argv)
        for arg in arg_iter:
            if arg in ["-D", "--define"]:
                k,v = next(arg_iter).split("=")
                self._definitions.update({k: eval(v)})

            elif arg in ["-d"]:
                import yaml
                path = next(arg_iter)
                item = None
                if "#" in path:
                    from urllib.parse import urlparse
                    fullpath = urlparse(path)
                    filename = fullpath.path
                    item = fullpath.fragment
                else:
                    filename = path
                with open(filename, "r") as f:
                    defs = yaml.load(f, Loader=yaml.Loader)
                if item:
                    defs = defs[item]
                self._definitions.update(defs)

            elif arg in ["-w", "--write"]:
                target_name = next(arg_iter)
                print(self._write(target_name, arg_iter))
            
            elif arg in ["-a", "--append"]:
                self.appendages.append(next(arg_iter))
                
            elif arg in ["-l", "--list"]:
                self._list(arg_iter)
            
            elif arg in ["-e", "--exec"]:
                target_name = next(arg_iter)
                self._exec(target_name, arg_iter)
            
            elif arg in ["-h", "--help"]:
                self._usage(arg_iter)


    

    def _list(self, arg_iter):
        items = "all"
        for arg in arg_iter:
            items = arg

        if items in ["proc", "all"]:
            print("Procedures:")
            for k,v in self._procedures.items():
                print(f"\t{k:20}\t{inspect.signature(v)}")
            print("\n")
        
        if items in ["names", "all"]:
            print("Objects:")
            for k,v in self._objects.items():
                print(f"\t{k:20}\t{v}")
            print("\n")


    def _write(self, target_name, arg_iter):
        from anabel.writers import OpenSeesWriter, FEDEAS_Writer

        target = Path(target_name)
        ext = target.suffix
        appends = "\n"

        if ext:
            fmt = {".tcl": OpenSeesWriter, ".m": FEDEAS_Writer}[ext]
        else:
            fmt = OpenSeesWriter

        for fn in self.appendages:
            with open(fn, "r") as f:
                appends = "\n".join((appends,f.read()))


        return self._objects[target.stem].dump(fmt, definitions=self._definitions) + appends

    def _exec(self, target_name, arg_iter):
        kwds = {}
        target = Path(target_name)
        ext = target.suffix
        if ext:
            prgm = {".tcl": "opensees", ".m": "octave"}[ext]
        else:
            prgm = None

        positional_args = []
        for arg in arg_iter:
            if arg == "-s":
                value = next(arg_iter)
                if ":" in value:
                    k,v = value.split(":")
                    kwds.update({k: self._objects[v]})
                elif "=" in value:
                    k,v = value.split("=")
                    kwds.update({k: eval(v)})
            else:
                positional_args.append(arg)

        if prgm is None:
            print(self._procedures[target_name](*positional_args, **kwds))

        else:
            import subprocess
            print(subprocess.check_output([prgm], input=self._write(target_name, arg_iter).encode()).decode())
            
    def _usage(self, arg_iter):
        items = "all"
        for arg in arg_iter:
            items = arg

        print(f"""usage: {self._name} [Options] COMMAND TARGET

Targets can be either objects or procedures.

Commands:
-e/--exec PROCEDURE\tExecute the specified procedure.

-l/--list\t\tList available procedures and objects. 

-w/--write TARGET

-h/--help\t\tPrint this message and exit.

Parameters
-D PARAMETER=VALUE
-d PARAM_FILE\t\t YAML parameter file.

""")


