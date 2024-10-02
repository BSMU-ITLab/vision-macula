"""
See build instructions:
https://github.com/BSMU-ITLab/vision/blob/main/scripts/build.py
"""

from pathlib import Path

from bsmu.vision.app.builder import AppBuilder

import orgname.template.app
import orgname.template.plugins


if __name__ == '__main__':
    app_builder = AppBuilder(
        project_dir=Path(__file__).resolve().parents[1],
        script_path_relative_to_project_dir=Path('src/orgname/template/app/__main__.py'),

        app_name=orgname.template.app.__title__,
        app_version=orgname.template.app.__version__,
        app_description=orgname.template.app.__description__,
        icon_path_relative_to_project_dir=Path('src/orgname/template/app/images/icons/vision.ico'),

        add_packages=['orgname.template.app', 'orgname.template.plugins', 'scipy.optimize', 'scipy.integrate'],
        add_packages_with_data=[orgname.template.app, orgname.template.plugins],
    )
    app_builder.build()
