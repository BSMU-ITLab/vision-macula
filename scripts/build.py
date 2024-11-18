"""
See build instructions:
https://github.com/BSMU-ITLab/vision/blob/main/scripts/build.py
"""

from pathlib import Path

from bsmu.vision.app.builder import AppBuilder

import bsmu.macula
from bsmu.macula.app import MaculaApp


if __name__ == '__main__':
    app_builder = AppBuilder(
        project_dir=Path(__file__).resolve().parents[1],
        app_class=MaculaApp,

        script_path_relative_to_project_dir=Path('src/bsmu/macula/app/__main__.py'),
        icon_path_relative_to_project_dir=Path('src/bsmu/macula/app/images/icons/macula.ico'),

        add_packages=['bsmu.macula', 'scipy.optimize', 'scipy.integrate'],
        add_packages_with_data=[bsmu.macula],
    )
    app_builder.build()
