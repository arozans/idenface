from pathlib import Path

import pytest


@pytest.fixture
def unpatched_home_dir(mocker):
    return mocker.patch('src.utils.filenames._get_home_directory', return_value=Path.home())
