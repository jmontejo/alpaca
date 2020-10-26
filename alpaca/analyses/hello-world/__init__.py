from alpaca.core import BaseMain
from alpaca.batch import BatchManager

def register_cli(subparser,parentparser):

    analysis_name = 'hello-world'
    analysis_defaults = {
        "Main"       : HelloWorld, #no quotes, pointer to the class
        "example"    : False,
    }

    # Create your own sub-command and add arguments
    parser = subparser.add_parser(analysis_name, parents=[parentparser],
                                   help='Hello world sub-command.')
    parser.add_argument('--example', action='store_true',
                        help='example argument')

    return analysis_defaults


class HelloWorld(BaseMain):

    pass

class BatchManagerHelloWorld(BatchManager):

    pass
