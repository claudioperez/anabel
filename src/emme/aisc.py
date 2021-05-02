import functools
import elle.sections


@functools.wraps(elle.sections.load_aisc)
def load(name: str, properties: str = "A, I"):
    return elle.sections.load_aisc(name, properties)
