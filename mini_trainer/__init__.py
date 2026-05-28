# noqa: D104
from argparse import RawTextHelpFormatter

from tqdm.auto import tqdm

from mini_trainer.utils import get_rank


class TQDM(tqdm):
    """Wrapper around tqdm.auto.tqdm that automatically disables output on non-zero DDP ranks."""

    def __new__(cls, *args, **kwargs):
        if get_rank() > 0:
            kwargs["disable"] = True
        return super().__new__(cls, *args, **kwargs)


class Formatter(RawTextHelpFormatter):  # noqa: D101
    # only change how the “invocation” is rendered
    def _format_action_invocation(self, action):
        # for option-style args, join the option strings and drop the metavar entirely
        if action.option_strings:
            return ", ".join(action.option_strings)
        # otherwise (positional args), fall back to the default
        return super()._format_action_invocation(action)
