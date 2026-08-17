"""Unified command-line entry point."""

from importlib import import_module
import sys


COMMANDS = {
    ("data", "prepare"): "mllms.data.tinystories",
    ("data", "prepare-babylm"): "mllms.data.babylm",
    ("data", "prepare-wikipedia"): "mllms.data.wikipedia",
    ("data", "tokenize"): "mllms.tokenizer.tokenize",
    ("tokenizer", "train"): "mllms.tokenizer.train",
    ("train",): "mllms.training.cli",
    ("blimp", "download"): "mllms.evaluation.blimp.download",
    ("blimp", "prepare"): "mllms.evaluation.blimp.prepare",
    ("blimp", "evaluate"): "mllms.evaluation.blimp.evaluate",
    ("analyze", "prepare-sva"): "mllms.evaluation.sva.prepare",
    ("analyze", "build-controlled-sva"): "mllms.evaluation.sva.controlled",
    ("analyze", "evaluate-sva"): "mllms.evaluation.sva.evaluate",
    (
        "analyze",
        "activation-patching",
    ): "mllms.interpretability.activation_patching.runner",
    ("analyze", "sanity-check"): "mllms.evaluation.sanity",
    ("plot",): "mllms.visualization.cli",
}

HELP = """Usage: mllms COMMAND [ARGS]

Commands:
  data prepare                    Prepare fixed TinyStories splits
  data prepare-babylm             Prepare official BabyLM 100M/dev/test
  data prepare-wikipedia          Prepare English Wikipedia splits
  tokenizer train                 Train the SentencePiece BPE tokenizer
  data tokenize                   Create uint16 token streams
  train                           Train or resume the cloned-language GPT
  blimp download                  Download BLiMP agreement data
  blimp prepare                   Tokenize BLiMP pairs
  blimp evaluate                  Evaluate a checkpoint on BLiMP
  analyze prepare-sva             Build controlled CausalGym SVA pairs
  analyze build-controlled-sva    Build lexical-matched SVA minimal pairs
  analyze evaluate-sva            Evaluate SVA and select sanity pairs
  analyze activation-patching     Patch all layer/token/head sites
  analyze sanity-check            Inspect checkpoint predictions and loss
  plot training|blimp|sva|patching
                                  Create figures and CSV summary tables
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
