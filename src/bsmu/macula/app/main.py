from bsmu.vision.app import App

from bsmu.macula.app import __title__, __version__


class MaculaApp(App):
    pass


def run_app():
    app = MaculaApp(__title__, __version__)
    app.run()


if __name__ == '__main__':
    run_app()
