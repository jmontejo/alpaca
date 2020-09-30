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
    subparser = parser.add_subparsers(title='analyses commands',
                                      help='sub-command help')

    discovered_plugins = {
        name: importlib.import_module(name) for finder, name, ispkg in iter_namespace(alpaca.analyses)
    }

    for a,b in discovered_plugins.items():
        b.register_cli(subparser)

    args = parser.parse_args()


    main = args.Main(args)
    main.run()