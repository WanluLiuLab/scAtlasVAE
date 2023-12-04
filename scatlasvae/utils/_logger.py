import atexit
import time
from functools import reduce
import tqdm

verbosity = 1
""" Global verbosity level, choose from {0,...,6}. """

class Colors:
    BLACK   = "\033[0;30m"
    DGRAY   = "\033[1;30m"
    RED     = "\033[0;31m"
    LRED    = "\033[1;31m"
    GREEN   = "\033[0;32m"
    LGREEN  = "\033[1;32m"
    ORANGE  = "\033[0;33m"
    YELLOW  = "\033[1;33m"
    BLUE    = "\033[0;34m"
    LBLUE   = "\033[1;34m"
    PURPLE  = "\033[0;35m"
    LPURPLE = "\033[1;35m"
    CYAN    = "\033[0;36m"
    LCYAN   = "\033[1;36m"
    LGRAY   = "\033[0;37m"
    WHITE   = "\033[1;37m"
    BRED    = "\033[0;37;41m"
    BGREEN  = "\033[0;37;42m"
    BYELLOW = "\033[0;37;43m"
    BBLUE   = "\033[0;37;44m"
    NC      = "\033[0m"



def filename(filename=''):
    """ 
    @brief Define filename of logfile. 
           If not defined, log output will be to the standard output.

    @arg filename : str
    """
    global logfilename
    logfilename = filename
    # if providing a logfile name, automatically set verbosity to a very high level
    verbosity(5)

def m(v=0,*msg):
    """ 
    @brief Write message to log output, depending on verbosity level.

    @arg v : int
        Verbosity level of message.
    @arg *msg : 
        One or more arguments to be formatted as string. Same behavior as print
        function.
    """
    if verbosity > v:
        print(time.asctime(), end = '\t')
        mi(*msg)


def mi(*msg):
    """ 
    @brief Write message to log output, ignoring the verbosity level.

    @arg *msg : 
        One or more arguments to be formatted as string. Same behavior as print
        function.
    """
    if logfilename == '':
        # in python 3, the following works
        # print(*msg)
        # due to compatibility with the print statement in python 2 we choose
        print(' '.join([str(m) for m in msg]))
    else:
        out = ''
        for s in msg:
            out += str(s) + ' '
        with file(filename, 'a') as f:
            f.write(out + '\n')

def mt(*msg):
    """ 
    @brief Write message to log output, ignoring the verbosity level.

    @brief *msg : 
        One or more arguments to be formatted as string. Same behavior as print
        function.
    """
    if logfilename == '':
        # in python 3, the following works
        # print(*msg)
        # due to compatibility with the print statement in python 2 we choose
        print(time.asctime() + ' - TDI - INFO - ', end = '\t')
        print(' '.join([str(m) for m in msg]))
    else:
        out = ''
        for s in msg:
            out += str(s) + ' '
        with file(filename, 'a') as f:
            f.write(time.asctime() + '\t')
            f.write(out + '\n')

def mw(*msg):
    """ 
    @brief Write message to log output, ignoring the verbosity level.

    @brief *msg : 
        One or more arguments to be formatted as string. Same behavior as print
        function.
    """
    if logfilename == '':
        # in python 3, the following works
        # print(*msg)
        # due to compatibility with the print statement in python 2 we choose
        print(time.asctime() + ' - TDI - WARNING - ', end = '\t')
        print(Colors.YELLOW + ' '.join([str(m) for m in msg]) + Colors.NC)
    else:
        out = ''
        for s in msg:
            out += str(s) + ' '
        with file(filename, 'a') as f:
            f.write(time.asctime() + '\t')
            f.write(out + '\n')

def _sec_to_str(t):
    """ 
    @brief Format time in seconds.

    @arg t : int
        Time in seconds.
    """
    return "%d:%02d:%02d.%03d" % \
        reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],
            [(t*1000,),1000,60,60])


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def get_tqdm():
    if is_notebook():
        from tqdm.autonotebook import tqdm as tqdm_notebook
        return tqdm_notebook
    else:
        return tqdm.tqdm

def _terminate():
    """ 
    @brief  Function called when program terminates.
            Similar to mt, but writes total runtime.
    """
    if verbosity > 0 and not is_notebook():
        now = time.time()
        elapsed_since_start = now - start
        # mt(_sec_to_str(elapsed_since_start),'- total runtime')


# further global variables
start = time.time() # time.time() is deprecated since version python version 3.3
intermediate = start
logfilename = ''
separator = 40*"-"

atexit.register(_terminate)
