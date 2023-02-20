import numpy as np


class Operator:
    def write(self, *args):
        return self.formula.format(*args)


class Add(Operator):
    def __init__(self):
        self.formula = '({} + {})'
        self.__name__ = 'add'
        self.sign = '+'

    def __call__(self, x, y):
        return x + y


class Sub(Operator):
    def __init__(self):
        self.formula = '({} - {})'
        self.__name__ = 'sub'
        self.sign = '-'

    def __call__(self, x, y):
        return x - y


class Add3(Operator):
    def __init__(self):
        self.formula = '({} + {} + {})'
        self.__name__ = 'add3'
        self.sign = '+'

    def __call__(self, x, y, z):
        return x + y + z


class Dif4(Operator):
    def __init__(self):
        self.formula = '({} - {} - {} - {})'
        self.__name__ = 'dif4'
        self.sign = '-'

    def __call__(self, x, y, z, t):
        return x - y - z - t


class Neg(Operator):
    def __init__(self):
        self.formula = '-{}'
        self.__name__ = 'neg'
        self.sign = '-'

    def __call__(self, x):
        return -x


class Mul(Operator):
    def __init__(self):
        self.formula = '({} * {})'
        self.__name__ = 'mul'
        self.sign = '*'

    def __call__(self, x, y):
        return x * y


class Cos(Operator):
    def __init__(self):
        self.formula = 'cos({})'
        self.__name__ = 'cos'
        self.sign = 'cos'

    def __call__(self, x):
        return np.cos(x)


class Sin(Operator):
    def __init__(self):
        self.formula = 'sin({})'
        self.__name__ = 'sin'
        self.sign = 'sin'

    def __call__(self, x):
        return np.sin(x)


class Pow(Operator):
    def __init__(self):
        self.formula = '({}**{})'
        self.__name__ = 'pow'
        self.sign = '**'

    def __call__(self, x, y):
        # проверка если степень от 0 до 1, то под степенью должно быть положительное иначе
        x = np.clip(x, -10, 10)
        y = np.clip(y, -10, 10)

        return np.abs(x)**np.abs(y)


class Sqrt(Operator):
    def __init__(self):
        self.formula = 'sqrt({})'
        self.__name__ = 'sqrt'
        self.sign = 'sqrt'

    def __call__(self, x):
        return np.sqrt(np.abs(x))


class Exp(Operator):
    def __init__(self):
        self.formula = 'np.exp({})'
        self.__name__ = 'exp'
        self.sign = 'exp'

    def __call__(self, x):
        x = np.clip(x, -np.inf, 1.7976931348623157e+308)
        return np.exp(x)


class Div(Operator):
    def __init__(self):
        self.formula = '({}/{})'
        self.__name__ = 'div'
        self.sign = '/'

    def __call__(self, x, y):

        if type(y) == np.ndarray:
            mask = y == 0
            y[mask] = 1
        else:
            y = 1
        return x/y
