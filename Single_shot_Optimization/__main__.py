import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


def run():
    commands = ["utils"]
    parser = ArgumentParser(
        description="Single shot optimization utilities",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("command", choices=commands)

    pos = len(sys.argv)
    for cmd in commands:
        if cmd in sys.argv:
            pos = sys.argv.index(cmd) + 1

    args = parser.parse_args(sys.argv[1:pos])
    if args.command == "utils":
        import utils

        utils.run(sys.argv[pos:])
    else:
        parser.print_help()


if __name__ == "__main__":
    run()
