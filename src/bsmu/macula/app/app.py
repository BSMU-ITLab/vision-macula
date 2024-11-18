from bsmu.vision.app import App

from bsmu.macula import __title__, __version__


class MaculaApp(App):
    TITLE = __title__
    VERSION = __version__
