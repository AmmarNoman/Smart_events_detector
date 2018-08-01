from app.plugin import AppPlugin
from .src.views import smartEventsDetector

__plugin__ = "SmartEventsDetector"
__version__ = "1.0.1"


class SmartEventsDetector(AppPlugin):

    def setup(self):
        self.register_blueprint(smartEventsDetector)