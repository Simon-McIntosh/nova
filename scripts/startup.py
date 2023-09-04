"""Configure python environment."""

import matplotlib as mpl
import numpy as np
import pandas
import seaborn as sns
import warnings


class Startup:
    """Manage environment settings."""

    def __init__(self):
        """Configure environment."""
        self.pandas()
        self.seaborn()
        warnings.filterwarnings("ignore", "", ResourceWarning)

    @staticmethod
    def seaborn():
        """Set seaborn context."""
        sns.set_context("talk")
        mpl.rcParams["figure.figsize"] = np.array([5, 3.5]) / 0.394

    @staticmethod
    def pandas():
        """Set pandas options."""
        options = {
            "display": {
                "max_columns": 6,
                "max_colwidth": 8,
                "expand_frame_repr": False,  # Don't wrap to multiple pages
                "max_rows": 16,
                "max_seq_items": 50,  # Max length of printed sequence
                "precision": 2,
                "show_dimensions": False,
            },
            "mode": {"chained_assignment": "warn"},  # Control SettingWithCopyWarning
            # "future": {"infer_string": True},
        }

        for category, option in options.items():
            for op, value in option.items():
                pandas.set_option(f"{category}.{op}", value)  # Python 3.6+


if __name__ == "__main__":
    Startup()
    del Startup  # Clean up namespace in the interpreter
