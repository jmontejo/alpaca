from alpaca.core import BaseMain

def register_cli(subparser):
    # Create your own sub-command and add arguments
    parser = subparser.add_parser('hello-world',
                                   help='Hello world sub-command.')
    parser.add_argument('--example', action='store_true',
                        help='example argument')

    # Set the function corresponding to your subcommand
    parser.set_defaults(Main=HelloWorld)

    return parser


class HelloWorld(BaseMain):

    pass
