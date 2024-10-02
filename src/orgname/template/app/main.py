from bsmu.vision.app import App

from orgname.template.app import __title__, __version__


class TemplateApp(App):
    pass


def run_app():
    app = TemplateApp(__title__, __version__)
    app.run()


if __name__ == '__main__':
    run_app()
