class Data:
    def __init__(self, doc, label):
        self.doc = doc
        self.label = label

    def setlabel(self, newlabel):
        self.label = newlabel

    def tostring(self):
        return self.label, self.doc