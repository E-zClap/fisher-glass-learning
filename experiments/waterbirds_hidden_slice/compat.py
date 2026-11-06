"""Small standard-library compatibility helpers for Python 3.8+.

The cluster environment used for the Waterbirds run may predate
``argparse.BooleanOptionalAction`` (added in Python 3.9).  Keep the command-line
surface identical on older interpreters so launchers can always pass both
``--flag`` and ``--no-flag`` forms.
"""

from __future__ import annotations

import argparse


if hasattr(argparse, "BooleanOptionalAction"):
    BooleanOptionalAction = argparse.BooleanOptionalAction
else:

    class BooleanOptionalAction(argparse.Action):
        """Python 3.8 backport of :class:`argparse.BooleanOptionalAction`."""

        def __init__(
            self,
            option_strings,
            dest,
            default=None,
            type=None,
            choices=None,
            required=False,
            help=None,
            metavar=None,
        ):
            expanded_options = []
            for option in option_strings:
                expanded_options.append(option)
                if option.startswith("--"):
                    expanded_options.append(f"--no-{option[2:]}")
            super().__init__(
                option_strings=expanded_options,
                dest=dest,
                nargs=0,
                default=default,
                type=type,
                choices=choices,
                required=required,
                help=help,
                metavar=metavar,
            )

        def __call__(self, parser, namespace, values, option_string=None):
            enabled = not (option_string or "").startswith("--no-")
            setattr(namespace, self.dest, enabled)

        def format_usage(self):
            return " | ".join(self.option_strings)
