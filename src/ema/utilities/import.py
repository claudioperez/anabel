import tensorflow as tf 

class ElemGenerator:
    def __init__(self, ElemData: dict):
        self.ElemData = ElemData

    def generate_ElemData(self):
        # generates a dictionary between a person and all the photos of that person
        ElemData = dict(self.ElemData)
        return ElemData


