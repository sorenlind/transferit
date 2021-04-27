"""CLI entry point."""
# pylint:disable=import-outside-toplevel

from pathlib import Path
from typing import Any

from .cli import command, create_parser
from .exceptions import TransferitError
from .version import VERSION

MAINPARSER_HELP = "print current version number"
SUBPARSERS_HELP = "%(prog)s must be called with a command:"
DESCRIPTION = """Description"""


def main():
    """Handle CLI arguments."""
    print("Transferit\n")
    create_parser(
        DESCRIPTION,
        VERSION,
        [_create_class_parser, _split_parser, _train_parser, _wrap_parser],
    )


def _create_class_parser(subparsers: Any):
    """Create parser for the 'create-class' command."""
    parser = subparsers.add_parser(
        "create-class",
        help="Create a class consisting of all images in specified folder.",
    )
    parser.add_argument(
        "SOURCE_FOLDER",
        type=Path,
        help="Folder containing images. Sub folders will also be included.",
    )

    parser.add_argument(
        "DEST_FOLDER",
        type=Path,
        help="Destination in which to store the images.",
    )

    parser.add_argument(
        "--n-max",
        type=int,
        help="Optional max number of images to include. Images are chosen at random.",
    )

    parser.set_defaults(func=_create_class)
    return parser


@command("Successfully completed training", "Training failed")
def _create_class(_, args: Any):
    from .preparation import create_class

    if not args.SOURCE_FOLDER.is_dir():
        raise TransferitError("Source folder does not exist.")

    if args.DEST_FOLDER.is_dir():
        raise TransferitError("Destination folder already exists.")
    args.DEST_FOLDER.mkdir(parents=True, exist_ok=False)

    create_class(args.SOURCE_FOLDER, args.DEST_FOLDER, (256, 256), args.n_max, 42)


def _split_parser(subparsers: Any):
    """Create parser for the 'split' command."""
    parser = subparsers.add_parser(
        "split",
        help="Create train / dev split for dataset.",
    )
    parser.add_argument(
        "SOURCE_FOLDER",
        type=Path,
        help="Folder containing images. Folder must have sub folders for the classes .",
    )

    parser.add_argument(
        "DEST_FOLDER",
        type=Path,
        help="Destination folder. A 'train' and a 'dev' folder will be created here.",
    )

    parser.set_defaults(func=_split_data)
    return parser


@command("Successfully created train / dev split", "Creating split failed")
def _split_data(_, args: Any):
    from .preparation import split_data

    if not args.SOURCE_FOLDER.is_dir():
        raise TransferitError("Source folder does not exist.")

    train_folder = args.DEST_FOLDER / "train"
    if train_folder.is_dir():
        raise TransferitError("Destination train folder already exists.")

    dev_folder = args.DEST_FOLDER / "dev"
    if dev_folder.is_dir():
        raise TransferitError("Destination dev folder already exists.")

    split_data(args.SOURCE_FOLDER, train_folder, dev_folder, 42, 0.1)


def _train_parser(subparsers: Any):
    """Create parser for the 'train' command."""
    parser = subparsers.add_parser(
        "train",
        help="Train model",
    )
    parser.add_argument(
        "TRAIN_FOLDER",
        type=Path,
        help="Folder containing training files.",
    )

    parser.add_argument(
        "DEV_FOLDER",
        type=Path,
        help="Folder containing dev files.",
    )

    parser.add_argument(
        "MODEL_FOLDER",
        type=Path,
        help="Folder in which to store the model.",
    )

    parser.set_defaults(func=_train)
    return parser


@command("Successfully completed training", "Training failed")
def _train(_, args: Any):
    from .training import train

    if not args.TRAIN_FOLDER.is_dir():
        raise TransferitError("Train folder does not exist.")

    if not args.DEV_FOLDER.is_dir():
        raise TransferitError("Dev folder does not exist.")

    if args.MODEL_FOLDER.is_dir():
        raise TransferitError("Model folder already exists.")

    train(args.TRAIN_FOLDER, args.DEV_FOLDER, args.MODEL_FOLDER, (256, 256))


def _wrap_parser(subparsers: Any):
    """Create parser for the 'wrap' command."""
    parser = subparsers.add_parser(
        "wrap",
        help="Wrap a trained model up to be served with TF Serving",
    )
    parser.add_argument(
        "MODEL_FILE",
        type=Path,
        help="Trained model (hdf5 file).",
    )

    parser.add_argument(
        "OUTPUT_FOLDER",
        type=Path,
        help="Folder in which to store the wrapped model.",
    )

    parser.add_argument(
        "-c",
        "--class-names",
        nargs="+",
        help="Class names (must adhere to order defined by model)",
        required=True,
    )

    parser.set_defaults(func=_wrap)
    return parser


@command("Successfully wrapped model", "Wrapping failed")
def _wrap(_, args: Any):
    from .wrapping import wrap_model

    if not args.MODEL_FILE.is_file():
        raise TransferitError("Model file does not exist.")

    if args.OUTPUT_FOLDER.is_dir():
        raise TransferitError("Output folder already exists.")

    args.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=False)

    wrap_model(
        model_file=args.MODEL_FILE,
        output_folder=args.OUTPUT_FOLDER,
        class_names=args.class_names,
        top_k=len(args.class_names),
    )


if __name__ == "__main__":
    main()
