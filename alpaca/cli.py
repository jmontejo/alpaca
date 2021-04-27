import logging

import importlib
import pkgutil
from pathlib import Path

from progressbar import progressbar
from collections import namedtuple

import alpaca.analyses
import alpaca.log

__all__ = ['cli']


logging.getLogger('matplotlib').setLevel(logging.WARNING)

log = logging.getLogger('alpaca')

NN_args = namedtuple('NN_args', ['jets', 'zero_jets','extra_jet_fields','extras','outputs','categories','scalars'])
Run_args = namedtuple('Run_args',['debug','train','tag','input_files','input_categories','shuffle_events','shuffle_jets','fast', 'test_sample','label_roc','fflayers','ncombos','per_jet','simple_nn','cola_lola','hydra','spectators'])

def iter_namespace(ns_pkg):
    # Specifying the second argument (prefix) to iter_modules makes the
    # returned name an absolute name instead of a relative one. This allows
    # import_module to work without having to do additional modification to
    # the name.
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")


def split_args(args):
    def digest_accepted(args,_class):
        print(args.__dict__)
        accept = {k:v for k,v in args.__dict__.items() if k in _class._fields}
        nt = _class(**accept)
        for k in accept: del args.__dict__[k]
        return nt

    nnargs  = digest_accepted(args, NN_args) 
    runargs = digest_accepted(args, Run_args)
    return nnargs, runargs, args

def cli():
    import argparse
    class UniqueAppend(argparse.Action):
        """argparse.Action subclass to store distinct values"""
        def __call__(self, parser, namespace, values, option_string=None):
            try:
                getattr(namespace,self.dest).add( values )
            except AttributeError:
                setattr(namespace,self.dest,set([values]))


    sharedparser = argparse.ArgumentParser(description='Dummy argparser for shared arguments',add_help=False)

    parser = argparse.ArgumentParser(description='ML top-like tagger.')
    sharedparser.add_argument('--debug', action='store_true', help='Debug verbosity')
    sharedparser.add_argument('--train', '-t', action='store_true', help='Run training')
    sharedparser.add_argument('--write-output', '-w', action='store_true', help='Stores the result of the evaluation on the test sample')
    sharedparser.add_argument('--output-dir', type=Path, default=Path('data'),
                        help='path to the output directory')
    sharedparser.add_argument('--tag', default='alpaca',
                        help='tag the output')

    sharedparser.add_argument('--input-files', '-i', required=True, type=Path,
                        action='append',
                        help='path to the file with the input events')
    sharedparser.add_argument('--input-categories', '-ic', type=int,
                        action='append',
                        help='Category to assign to each of the input files', default=[])
    sharedparser.add_argument('--shuffle-events', action='store_true')
    sharedparser.add_argument('--shuffle-jets', action='store_true')
    sharedparser.add_argument('--fast', action='store_true',help="Run only over sqrt(N) events for a fast test")
    sharedparser.add_argument('--test-sample', type=int, default=-1, help="How many events to use for the test sample. If running the training, the training sample is all the remaining events. If negative, use the whole sample")
    sharedparser.add_argument('--label-roc', type=str, default="", help="Label added to the name of the ROC curve plots")

    #needed to define the NN, we need to know them to apply the NN on a sample
    sharedparser.add_argument('--jets', help='Number of jets to be used', type=int)
    sharedparser.add_argument('--zero-jets', help='Number of jet positions that can be left empty', type=int)
    sharedparser.add_argument('--extra-jet-fields', help='Additional information to be included with the jets', action=UniqueAppend, default=[])
    sharedparser.add_argument('--extras', help='Number of extra objects to be used', type=int, default=0)
    sharedparser.add_argument('--outputs', help='Number of output flags. Comma-separated list with length equal to the number of categories. \
                                           Can use "N" to read the number of jets. E.g. "N,5,6"')
    sharedparser.add_argument('--categories', help='Number of categories to consider, or comma-separated list of category names')
    sharedparser.add_argument('--scalars', help='List of scalar variables to be used in the training', action=UniqueAppend, default=[])

    #needed to define the NN, embedded in the architecture but we don't need to know them after training
    sharedparser.add_argument("--fflayers", nargs="+", type=int, default=[200])
    sharedparser.add_argument("--ncombos", type=int, default=30)
    sharedparser.add_argument('--per-jet', action='store_true', help='Run CoLaLola in per-jet mode')
    nnchoice = sharedparser.add_mutually_exclusive_group()
    nnchoice.add_argument("--simple-nn",action="store_true")
    nnchoice.add_argument("--cola-lola",action="store_true")
    nnchoice.add_argument('--hydra', action='store_true')
    sharedparser.add_argument('--spectators', help='List of spectator variables with the same name as in the input df', action=UniqueAppend, default=[])


    subparser = parser.add_subparsers(title='analyses commands', dest='subparser')

    discovered_plugins = {
        name: importlib.import_module(name) for finder, name, ispkg in iter_namespace(alpaca.analyses)
    }

    analysis_defaults = {}
    for a,b in discovered_plugins.items():
        sub  = list(iter_namespace(b))
        print (a,b,sub)
        if sub:
            for _1,name,_2 in sub :
                c = importlib.import_module(name)
                x = c.register_cli(subparser,sharedparser)
                analysis_defaults.update( dict((x,) ))
        else:
            x = b.register_cli(subparser,sharedparser)
            analysis_defaults.update( dict((x,) ))

    chosensubparser = parser.parse_args().subparser
    parser.set_defaults(**analysis_defaults[chosensubparser])
    sharedparser.set_defaults(**analysis_defaults[chosensubparser])
    args = parser.parse_args()

    if "," in args.outputs:
        args.outputs = [int(x.lower().replace("n",str(args.jets))) for x in args.outputs.split(",")]
    else:
        c = int(args.outputs)
        args.outputs =  [1]*c


    args.totaloutputs = sum(args.outputs)
    args.nscalars = len(args.scalars)
    args.nextrafields = len(args.extra_jet_fields)

    try:
        c = int(args.categories)
        args.categories = [ 'category_%d'%i for i in range(c)]
    except ValueError:
        args.categories = args.categories.split(",")
    finally:
        assert len(args.outputs) == len(args.categories), "Output flags and categories don't match: %r %r"%(args.outputs, args.categories)
    args.ncategories = len(args.categories)

    print(args)
    #nnargs, runargs, userargs = split_args(args)
    #main = args.Main(nnargs, runargs, userargs)
    main = args.Main(args)
    main.run()
    #main.plots()
