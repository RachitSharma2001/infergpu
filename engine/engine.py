
class Engine:
    def __init__(self, model_name):
        self.model_name = model_name

    def stream(self, prompt):
        yield "I"
        yield "am"
        yield "rachit"

    