"""Internal torchrun module carrying an action default below user configuration."""

import sys


def main():
    action, *argv = sys.argv[1:]
    if action not in ("test", "test_all"):
        raise ValueError("Launcher inference action must be test or test_all")
    from inference import main as inference_main

    inference_main(argv, action=action)


if __name__ == "__main__":
    main()
