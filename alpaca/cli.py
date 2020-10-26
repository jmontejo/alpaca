import logging

import importlib
import pkgutil
from pathlib import Path

from progressbar import progressbar

import alpaca.analyses
import alpaca.log

__all__ = ['cli']


logging.getLogger('matplotlib').setLevel(logging.WARNING)

log = logging.getLogger('alpaca')


def iter_namespace(ns_pkg):
    # Specifying the second argument (prefix) to iter_modules makes the
    # returned name an absolute name instead of a relative one. This allows
    # import_module to work without having to do additional modification to
    # the name.
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")



def cli():
    import argparse
    class UniqueAppend(argparse.Action):
        """argparse.Action subclass to store distinct values"""
        def __call__(self, parser, namespace, values, option_string=None):
            try:
                getattr(namespace,self.dest).add( values )
            except AttributeError:
                setattr(namespace,self.dest,set([values]))

    sharedparser = argparse.ArgumentParser(description='Dummy argparser for shared arguments',add_help=False, argument_default=argparse.SUPPRESS)

    parser = argparse.ArgumentParser(description='ML top-like tagger.')
    sharedparser.add_argument('--debug', action='store_true', help='Debug verbosity')
    sharedparser.add_argument('--output-dir', type=Path, default=Path('data'),
                        help='path to the output directory')
    sharedparser.add_argument('--tag', default='alpaca',
                        help='tag the output')
    sharedparser.add_argument('--jets', help='Number of jets to be used', type=int)
    sharedparser.add_argument('--extra-jet-fields', help='Additional information to be included with the jets', action=UniqueAppend, default=[])
    sharedparser.add_argument('--extras', help='Number of extra objects to be used', type=int, default=0)
    sharedparser.add_argument('--outputs', help='Number of output flags. Comma-separated list with length equal to the number of categories. \
                                           Can use "N" to read the number of jets. E.g. "N,5,6"')
    sharedparser.add_argument('--categories', help='Number of categories to consider, or comma-separated list of category names')
    sharedparser.add_argument('--scalars', help='List of scalar variables to be used in the training', action="append", default=[])
    sharedparser.add_argument('--analysis_defaults', help='List of scalar variables to be used in the training')

    subparser = parser.add_subparsers(title='analyses commands', dest='subparser')

    discovered_plugins = {
        name.split(".")[-1]: importlib.import_module(name) for finder, name, ispkg in iter_namespace(alpaca.analyses)
    }

    analysis_defaults = {}
    for a,b in discovered_plugins.items():
        analysis_defaults[a] = b.register_cli(subparser,sharedparser)

    chosensubparser = parser.parse_args().subparser
    parser.set_defaults(**analysis_defaults[chosensubparser])
    args = parser.parse_args()

    try:
        args.outputs = [int(x.lower().replace("n",str(args.jets))) for x in args.outputs.split(",")]
    except AttributeError:
        try: 
            c = int(args.outputs)
            args.outputs =  [c]
        except ValueError:
            pass

    args.totaloutputs = sum(args.outputs)
    args.nscalars = len(args.scalars)
    args.nextrafields = len(args.extra_jet_fields)

    try:
        c = int(args.categories)
        args.categories = [ 'category_%d'%i for i in range(c)]
    except ValueError:
        pass
    finally:
        assert len(args.outputs) == len(args.categories), "Output flags and categories don't match: %r %d"%(args.outputs, args.categories)

    main = args.Main(args)
    main.run()
    #main.plots()
