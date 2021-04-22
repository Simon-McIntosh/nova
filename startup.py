import pandas as pd

def start():
    options = {
        'display': {
            'max_columns': 8,
            'max_colwidth': 10,
            'expand_frame_repr': False,  # Don't wrap to multiple pages
            'max_rows': 14,
            'max_seq_items': 50,         # Max length of printed sequence
            'precision': 3,
            'show_dimensions': False
        },
        'mode': {
            'chained_assignment': 'warn'   # Controls SettingWithCopyWarning
        }
    }

    for category, option in options.items():
        for op, value in option.items():
            pd.set_option(f'{category}.{op}', value)  # Python 3.6+


if __name__ == '__main__':
    start()
    del start  # Clean up namespace in the interpreter
