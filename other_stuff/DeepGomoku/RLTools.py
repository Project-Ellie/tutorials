class State:
    def valid_actions(self):
        raise NotImplementeddError()
    def after(move):
        raise NotImplementedError()
        
        
class Action:
    pass


class QFunction:
    def __apply__(self, state, action):
        raise NotImplementedError()

        
class VFunction:
    def __apply__(self, state):
        raise NotImplementedError()
        
        
class Policy:
    def distr(self, state):
        raise NotImplementedError()
    def draw(self, state):
        raise NotImplementedError()


class Actor:
    def suggest_action(self, state, max_width, max_depth):
        raise NotImplementedError()