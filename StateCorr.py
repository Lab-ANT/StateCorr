class StateCorr():
    def __init__(self, segment_component) -> None:
        self.__segment_component = segment_component

    def fit_predict(self, X):
        self.__segment_component.fit(X)
        self.__segment_component

    def fit(self):
        pass

    def predict(self):
        pass