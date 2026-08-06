"""Unified command-line entry point."""

from importlib import import_module
import sys


COMMANDS = {
    ("data", "prepare"): "data_lib.prepare_tinystories",
    ("data", "tokenize"): "tokenizer.tokenization",
    ("tokenizer", "train"): "tokenizer.train_tokenizer",
    ("train",): "train",
    ("blimp", "download"): "blimp.download_blimp",
    ("blimp", "prepare"): "blimp.prepare_blimp",
    ("blimp", "evaluate"): "blimp.evaluate_blimp",
    ("analyze", "prepare-sva"): "analysis.prepare_sva_pairs",
    ("analyze", "activation-patching"): "analysis.activation_patching",
    ("plot",): "plots.reports",
}

HELP = """Usage: mllms COMMAND [ARGS]

Commands:
  data prepare                    Prepare fixed TinyStories splits
  tokenizer train                 Train the SentencePiece BPE tokenizer
  data tokenize                   Create uint16 token streams
  train                           Train or resume the cloned-language GPT
  blimp download                  Download BLiMP agreement data
  blimp prepare                   Tokenize BLiMP pairs
  blimp evaluate                  Evaluate a checkpoint on BLiMP
  analyze prepare-sva             Build aligned SVA pairs for patching
  analyze activation-patching     Run activation patching
  plot training|blimp|patching    Create figures and CSV summary tables
"""


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in {"-h", "--help"}:
        print(HELP)
        return

    for command, module_name in COMMANDS.items():
        if tuple(args[: len(command)]) == command:
            sys.argv = ["mllms " + " ".join(command), *args[len(command) :]]
            import_module(module_name).main()
            return

    print(HELP, file=sys.stderr)
    raise SystemExit(f"Unknown command: {' '.join(args[:2])}")


if __name__ == "__main__":
    main()
