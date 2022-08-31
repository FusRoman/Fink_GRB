"""
Usage:
    fink_grb start_gcn_stream [options]
    fink_grb init [options]
    fink_grb -h | --help
    fink_grb --version

Options:
  init                             initialise the environment for fink_grb.
  start_gcn_stream                 start to listen the gcn stream.
  -h --help                        Show help and quit.
  --version                        Show version.
  --config FILE                    Specify the config file.
  --verbose                        Print information and progress bar during the process.
"""

from docopt import docopt
import fink_grb

from fink_grb.online.gcn_stream import start_gcn_stream

from fink_grb.init import init_fink_grb

def main():

    # parse the command line and return options provided by the user.
    arguments = docopt(__doc__, version=fink_grb.__version__)


    if arguments["start_gcn_stream"]:

            start_gcn_stream(arguments)
    
    elif arguments["init"]:

        init_fink_grb(arguments)

        exit(0)

    else:
        exit(0)