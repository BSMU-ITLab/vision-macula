"""
See build instructions:
https://github.com/BSMU-ITLab/vision/blob/main/scripts/build.py
"""

from pathlib import Path

from bsmu.vision.app.builder import AppBuilder

import bsmu.macula.app
import bsmu.macula.plugins


if __name__ == '__main__':
    app_builder = AppBuilder(
        project_dir=Path(__file__).resolve().parents[1],
        script_path_relative_to_project_dir=Path('src/bsmu/macula/app/__main__.py'),

        app_name=bsmu.macula.app.__title__,
        app_version=bsmu.macula.app.__version__,
        app_description=bsmu.macula.app.__description__,
        icon_path_relative_to_project_dir=Path('src/bsmu/macula/app/images/icons/macula.ico'),

        add_packages=['bsmu.macula.app', 'bsmu.macula.plugins', 'scipy.optimize', 'scipy.integrate'],
        add_packages_with_data=[bsmu.macula.app, bsmu.macula.plugins],
    )
    app_builder.build()
