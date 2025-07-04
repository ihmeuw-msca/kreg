from collections import OrderedDict

import pandas as pd


class Logger:
    """
    table printer and convergence tracker.
      • Create once; call `log(iter=..., mu=..., ...)` each IPM step.
      • All rows are kept in `self.rows` (a list of OrderedDicts).
      • Call `to_dataframe()` at any point to get a pandas DataFrame.
    """

    _LINE_CHAR = "─"
    _COL_SEP = "│"

    def __init__(self, col_specs=None, verbose=True):
        if col_specs is None:
            col_specs = OrderedDict(
                [
                    ("iter", "{:>4d}"),
                    ("obj_val", "{:>10.3e}"),
                    ("gnorm_inf", "{:>11.2e}"),
                    ("gnorm2", "{:>10.2e}"),
                    ("Δx", "{:>7.1e}"),
                    ("step", "{:>6.1e}"),
                    ("time", "{:>6.2f}s"),
                    ("armijo_rat", "{:>10.2f}"),
                ]
            )
        if not isinstance(col_specs, OrderedDict):
            col_specs = OrderedDict(col_specs)

        self.col_specs = col_specs
        self._hdr_printed = False
        self._border = self._LINE_CHAR * (
            sum(self._col_widths()) + 3 * len(col_specs) + 1
        )

        self.rows: list[OrderedDict] = []
        self.verbose = verbose

    def log(self, **kwargs):
        """
        Print one formatted row *and* append it to self.rows.
        Missing keys show as blanks in the table and as None in storage.
        Extra keys are ignored in printing but kept in storage.
        """

        if self.verbose is True:
            if not self._hdr_printed:
                self._print_header()
                self._hdr_printed = True

            fmt_cells = []
            for key, fmt in self.col_specs.items():
                val = kwargs.get(key, "")
                cell = fmt.format(val) if val != "" else " " * self._width(fmt)
                fmt_cells.append(cell)
            row_str = (
                f"{self._COL_SEP} "
                + f" {self._COL_SEP} ".join(fmt_cells)
                + f" {self._COL_SEP}"
            )
            print(row_str)

        stored = OrderedDict()
        for key in self.col_specs:  # preserve column order
            stored[key] = kwargs.get(key, None)  # None if not supplied
        # also keep any extra diagnostics the solver included
        for k, v in kwargs.items():
            if k not in stored:
                stored[k] = v
        self.rows.append(stored)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the full history as a pandas DataFrame."""
        return pd.DataFrame(self.rows)

    def _print_header(self):
        hdr_cells = [
            f"{name:^{self._width(fmt)}}"
            for name, fmt in self.col_specs.items()
        ]
        header = (
            f"{self._COL_SEP} "
            + f" {self._COL_SEP} ".join(hdr_cells)
            + f" {self._COL_SEP}"
        )
        print(self._border)
        print(header)
        print(self._border)

    def _width(self, fmt: str) -> int:
        """Width of the formatted string produced by *fmt* for the dummy value 0."""
        return len(fmt.rstrip("s").format(0))

    def _col_widths(self):
        return (self._width(fmt) for fmt in self.col_specs.values())
