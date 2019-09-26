import pytest
import numpy as np

from edflow.debug import DebugDataset


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def test_debug_dset():
    D = DebugDataset(size=10)

    print(
        bcolors.WARNING
        + bcolors.BOLD
        + "Dear hacker,\nShould you see this message you probably have changed "
        "something in the DebugDataset code. Please consider that all "
        "dataset test depend on this piece of code, so you want to be very "
        "careful what you change.\n   Yours sincerely, jhaux" + bcolors.ENDC
    )

    assert len(D) == 10
    assert len(D.labels["label1"]) == 10

    assert D[0] == {"val": 0, "other": 0, "index_": 0}
    assert np.all(D.labels["label1"] == np.arange(10))
