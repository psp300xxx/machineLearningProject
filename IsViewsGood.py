from enum import Enum
class ViewsResult(Enum):
    POOR = 0
    MEDIUM = 1
    GOOD = 2
    @staticmethod
    def init(string):
        if string == "ViewsResult.POOR":
            return ViewsResult.POOR
        elif string == "ViewsResult.MEDIUM":
            return ViewsResult.MEDIUM
        elif string == "ViewsResult.GOOD":
            return ViewsResult.GOOD
        return None
