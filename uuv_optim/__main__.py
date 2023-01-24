import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from . import myring_cad


def run():
    parser = ArgumentParser(
        "UUV design optimization", formatter_class=ArgumentDefaultsHelpFormatter
    )

    commands = ["myring-cad", "cfd", "optimizers", "surrogates"]

    parser.add_argument(
        "command", choices=sorted(commands), help="The sub command to execute"
    )

    pos = len(sys.argv)

    for cmd in commands:
        if cmd in sys.argv:
            pos = sys.argv.index(cmd) + 1

    args = parser.parse_args(sys.argv[1:pos])

    if args.command == "myring-cad":
        myring_cad.run(sys.argv[pos:])
    else:
        parser.print_help()
