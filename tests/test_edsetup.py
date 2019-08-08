import subprocess
import os
import shutil

test_project = "test_project_delete_this_if_it_did_not_happen_automatically"


def test_edsetup_plus_edflow_compatibility():
    """Just make sure edsetup runs without errors and it is edflow compatible."""
    subprocess.run(
        f"edsetup -n {test_project}",
        shell=True,
        check=True,
    )
    test_config = os.path.join(test_project, "config.yaml")
    subprocess.run(
        f"edflow -t {test_config}",
        shell=True,
        check=True,
    )
    shutil.rmtree(test_project, ignore_errors=True)
