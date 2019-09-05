import sys, os, pdb


class ForkedPdb(pdb.Pdb):
    """Pdb subclass which works in subprocesses. We need to set stdin to be
    able to interact with the debugger. os.fdopen instead of
    open("/dev/stdin") keeps readline working.
    https://stackoverflow.com/a/31821795"""

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = os.fdopen(0)
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


fpdb = ForkedPdb()
# enable
# import edflow.fpdb as pdb; pdb.set_trace()
set_trace = fpdb.set_trace
