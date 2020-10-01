from pathlib import Path

from alpaca.core import BaseMain


def register_cli(subparser):
    # Create your own sub-command and add arguments
    parser = subparser.add_parser('elena',
                                   help='Hello world sub-command.')
    parser.add_argument('--example', action='store_true',
                        help='example argument')
    parser.add_argument('--input-files', '-i', required=True, type=Path,
                        nargs='+',
                        help='path to the file with the input events')
    parser.add_argument('--input-categories', '-ic', required=True, type=int,
                        nargs='+',
                        help='path to the file with the input events')

    # Set the function corresponding to your subcommand
    parser.set_defaults(Main=Elena)

    return parser


class Elena(BaseMain):

    pass
