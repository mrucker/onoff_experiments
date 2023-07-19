class PowerScheduler:
    def __init__(self, tzero:float, power:float):
        self.tzero = tzero
        self.power = power

    def __call__(self, t) -> float:
        return (1+t/self.tzero)**self.power
