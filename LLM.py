from abc import ABC, abstractmethod

class LLM(ABC):

    def __init__(self):
        super().__init__()
    # CHILD CLASSES MUST IMPLEMENT THESE ABSTRACT METHODS
        
    @abstractmethod
    def generate(self, prompt, tokens, batch):
        ''' Takes a prompt as a string and returns the output of the LLM as string'''
        pass

    @abstractmethod
    def cleanup(self):
        ''' Cleans up GPU and memory usage of the model '''
        pass
