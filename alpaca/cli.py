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
    parser = argparse.ArgumentParser(description='ML top-like tagger.')
    parser.add_argument('--debug', action='store_true', help='Debug verbosity')
    parser.add_argument('--output-dir', type=Path, default=Path('data'),
                        help='path to the output directory')
    parser.add_argument('--tag', default='alpaca',
                        help='tag the output')
    parser.add_argument('--jets', help='Number of jets to be used', type=int)
    parser.add_argument('--extra-jet-fields', help='Additional information to be included with the jets', action='append', default=[])
    parser.add_argument('--extras', help='Number of extra objects to be used', type=int, default=0)
    parser.add_argument('--outputs', help='Number of output flags. Comma-separated list with length equal to the number of categories. \
                                           Can use "N" to read the number of jets. E.g. "N,5,6"')
    parser.add_argument('--categories', help='Number of categories to consider', type=int)

    subparser = parser.add_subparsers(title='analyses commands',
                                      help='sub-command help')

    discovered_plugins = {
        name: importlib.import_module(name) for finder, name, ispkg in iter_namespace(alpaca.analyses)
    }

    for a,b in discovered_plugins.items():
        b.register_cli(subparser)

    args = parser.parse_args()
    args.outputs = [int(x.lower().replace("n",str(args.jets))) for x in args.outputs.split(",")]
    assert len(args.outputs) == args.categories, "Output flags and number of categories don't match: %r %d"%(args.outputs, args.categories)
    args.totaloutputs = sum(args.outputs)

    main = args.Main(args)
    main.run()
    #main.plots()