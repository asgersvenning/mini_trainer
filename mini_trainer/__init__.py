from tqdm import tqdm as TQDM # TODO: Context dependent progress bars

from argparse import RawTextHelpFormatter

class Formatter(RawTextHelpFormatter):
    # only change how the “invocation” is rendered
    def _format_action_invocation(self, action):
        # for option-style args, join the option strings and drop the metavar entirely
        if action.option_strings:
            return ', '.join(action.option_strings)
        # otherwise (positional args), fall back to the default
        return super()._format_action_invocation(action)