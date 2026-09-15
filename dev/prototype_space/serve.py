"""Compatibility entry point for the packaged prototype explorer."""

from mini_trainer.visualization.prototype_space.serve import *  # noqa: F403
from mini_trainer.visualization.prototype_space.serve import main

if __name__ == "__main__":
    main()
