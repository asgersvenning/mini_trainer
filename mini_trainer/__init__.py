from argparse import RawTextHelpFormatter

from tqdm import tqdm as TQDM  # # noqa: F401 TODO: Context dependent progress bars


class Formatter(RawTextHelpFormatter):
    # only change how the “invocation” is rendered
    def _format_action_invocation(self, action):
        # for option-style args, join the option strings and drop the metavar entirely
        if action.option_strings:
            return ', '.join(action.option_strings)
        # otherwise (positional args), fall back to the default
        return super()._format_action_invocation(action)