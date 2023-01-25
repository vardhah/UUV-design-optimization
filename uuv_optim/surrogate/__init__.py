from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from . import evaluate, train


def run(args):
    parser = ArgumentParser(
        "The drag surrogate model for UUV hulls",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    commands = ["train", "eval"]
    parser.add_argument("command", choices=sorted(commands))
    pos = len(args)

    for cmd in commands:
        if cmd in args:
            pos = args.index(cmd) + 1

    parsed_args = parser.parse_args(args[0:pos])

    if parsed_args.command == "train":
        train.run(args[pos:])
    elif parsed_args.command == "eval":
        evaluate.run(args[pos:])
    else:
        parser.print_help()
