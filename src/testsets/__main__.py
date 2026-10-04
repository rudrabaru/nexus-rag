"""
    python -m src.testsets generate --tenant T --set NAME [--count 50] [--model provider/model] [--dry-run] [--show-prompt]
    python -m src.testsets review   --tenant T --set NAME
    python -m src.testsets freeze   --tenant T --set NAME
    python -m src.testsets verify   --tenant T --set NAME
    python -m src.testsets list     --tenant T
    python -m src.testsets import   --tenant T --set NAME FILE.json     # a dataset file becomes a frozen set
    python -m src.testsets export   --tenant T --set NAME FILE.json     # a set's accepted questions as a file

Test sets live in Postgres (DATABASE_URL). generate reads the workspace's chunks and writes questions
into a draft set; review accepts, edits or rejects each; freeze makes the set immutable and hashes
it, and an experiment then uses it as "dataset": "testset:NAME". Ground truth is chunk ids, so a set
is only valid for the chunking it was made from: verify checks.
"""
import argparse
import sys

from src.runtime import ConfigurationError, bootstrap
from src.stores.testsets import TestSetError
from src.testsets import commands

COMMANDS = {
    "generate": commands.command_generate, "review": commands.command_review, "freeze": commands.command_freeze,
    "verify": commands.command_verify, "list": commands.command_list, "import": commands.command_import,
    "export": commands.command_export,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name, help_text, set_name=True, file=False):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--tenant", required=True, help="the workspace")
        if set_name:
            p.add_argument("--set", required=True, help="the test set's name")
        if file:
            p.add_argument("file")
        return p

    generate = command("generate", "write questions from the workspace's chunks into a draft set")
    generate.add_argument("--count", type=int, default=50)
    generate.add_argument("--model", help="provider/model (default: LLM_TESTSET)")
    generate.add_argument("--index", help="embedding index to read chunks from (default: the configured one)")
    generate.add_argument("--difficulties", default="easy,medium,hard", help="tiers, cycled in order")
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--min-interval", type=float, help=f"seconds between LLM calls (default per provider: {commands.PROVIDER_MIN_INTERVAL_SECONDS})")
    generate.add_argument("--dry-run", action="store_true", help="show what would be sampled; call no LLM")
    generate.add_argument("--show-prompt", action="store_true", help="print the first prompt")

    command("review", "accept, edit or reject each pending question")
    command("freeze", "make the set immutable and hash it, so an experiment can use it")
    command("verify", "check the accepted questions' source chunks exist in the workspace")
    command("list", "the workspace's test sets", set_name=False)
    command("import", "load a dataset file as a frozen test set", file=True)
    command("export", "write a test set's accepted questions as a dataset file", file=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        bootstrap("cli")
        return COMMANDS[args.command](args)
    except ConfigurationError as e:
        print(e)
        return 2
    except (TestSetError, ValueError) as e:
        print(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
