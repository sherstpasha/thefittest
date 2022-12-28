import numpy as np
import os

path = os.path.dirname(__file__) + '/shifts_data/'

fbias_data = np.loadtxt(path+'fbias_data.txt')
sphere_func_data = np.loadtxt(path+'sphere_func_data.txt')
high_cond_elliptic_rot_data = np.loadtxt(path
                                         + 'high_cond_elliptic_rot_data.txt')[:50]
elliptic_M_D2 = np.loadtxt(path+'elliptic_M_D2.txt')
elliptic_M_D10 = np.loadtxt(path+'elliptic_M_D10.txt')
elliptic_M_D30 = np.loadtxt(path+'elliptic_M_D30.txt')
elliptic_M_D50 = np.loadtxt(path+'elliptic_M_D50.txt')
schwefel_102_data = np.loadtxt(path+'schwefel_102_data.txt')
schwefel_206_data = np.loadtxt(path+'schwefel_206_data.txt')
o_206 = schwefel_206_data[0]
A_206 = schwefel_206_data[1:]
rosenbrock_func_data = np.loadtxt(path+'rosenbrock_func_data.txt')
rastrigin_func_data = np.loadtxt(path+'rastrigin_func_data.txt')[:50]
rastrigin_M_D2 = np.loadtxt(path+'rastrigin_M_D2.txt')
rastrigin_M_D10 = np.loadtxt(path+'rastrigin_M_D10.txt')
rastrigin_M_D30 = np.loadtxt(path+'rastrigin_M_D30.txt')
rastrigin_M_D50 = np.loadtxt(path+'rastrigin_M_D50.txt')
griewank_func_data = np.loadtxt(path+'griewank_func_data.txt')[:50]
griewank_M_D2 = np.loadtxt(path+'griewank_M_D2.txt')
griewank_M_D10 = np.loadtxt(path+'griewank_M_D10.txt')
griewank_M_D30 = np.loadtxt(path+'griewank_M_D30.txt')
griewank_M_D50 = np.loadtxt(path+'griewank_M_D50.txt')
ackley_M_D2 = np.loadtxt(path+'ackley_M_D2.txt')
ackley_M_D10 = np.loadtxt(path+'ackley_M_D10.txt')
ackley_M_D30 = np.loadtxt(path+'ackley_M_D30.txt')
ackley_M_D50 = np.loadtxt(path+'ackley_M_D50.txt')
ackley_func_data = np.loadtxt(path+'ackley_func_data.txt')[:50]
weierstrass_M_D2 = np.loadtxt(path+'weierstrass_M_D2.txt')
weierstrass_M_D10 = np.loadtxt(path+'weierstrass_M_D10.txt')
weierstrass_M_D30 = np.loadtxt(path+'weierstrass_M_D30.txt')
weierstrass_M_D50 = np.loadtxt(path+'weierstrass_M_D50.txt')
weierstrass_data = np.loadtxt(path+'weierstrass_data.txt')[:50]
schwefel_213_data = np.loadtxt(path+'schwefel_213_data.txt')
a_213 = schwefel_213_data[:100]
b_213 = schwefel_213_data[100:200]
alpha_213 = schwefel_213_data[-1]


class TestFunction:
    def __init__(self, global_optimum, fixed_accuracy, x_optimum):
        self.global_optimum = global_optimum
        self.fixed_accuracy = fixed_accuracy
        self.x_optimum = x_optimum

    def __call__(self, x):
        return self.f(x)

    def test(self):
        y = self(self.x_optimum.reshape(1, -1))
        return y - self.global_optimum < self.fixed_accuracy

    def build_grid(self, x, y):
        x1, y1 = np.meshgrid(x, y)
        xy = np.concatenate(
            [x1[:, :, np.newaxis], y1[:, :, np.newaxis]], axis=2)
        z = np.zeros(shape=xy.shape[:-1])
        for i, x_i in enumerate(xy):
            z[i] = self(x_i)
        return z


class TestShiftedFunction:
    def __init__(self, global_optimum, fixed_accuracy, x_optimum):
        self.global_optimum = global_optimum
        self.fixed_accuracy = fixed_accuracy
        self.x_optimum = x_optimum

    def shift(self, x):
        shape = x.shape
        axis = [1]*(len(shape)-1) + [-1]
        return x - self.x_optimum[:shape[-1]].reshape(axis)

    def __call__(self, x):
        z = self.shift(x)
        return self.f(z) + self.global_optimum


class TestShiftedRotatedFunction:
    def __init__(self, global_optimum, fixed_accuracy, x_optimum,
                 rotate_M_D2, rotate_M_D10, rotate_M_D30, rotate_M_D50):
        self.global_optimum = global_optimum
        self.fixed_accuracy = fixed_accuracy
        self.x_optimum = x_optimum
        self.rotate_M_D2 = rotate_M_D2
        self.rotate_M_D10 = rotate_M_D10
        self.rotate_M_D30 = rotate_M_D30
        self.rotate_M_D50 = rotate_M_D50

    def shift(self, x):
        shape = x.shape
        axis = [1]*(len(shape)-1) + [-1]
        return x - self.x_optimum[:shape[-1]].reshape(axis)

    def rotate(self, x):
        if x.shape[1] == 2:
            z = x@self.rotate_M_D2
        elif x.shape[1] == 10:
            z = x@self.rotate_M_D10
        elif x.shape[1] == 30:
            z = x@self.rotate_M_D30
        elif x.shape[1] == 50:
            z = x@self.rotate_M_D50
        return z

    def __call__(self, x):
        z = self.shift(x)
        z_rotated = self.rotate(z)
        return self.f(z_rotated) + self.global_optimum


# Test Functions
class OneMin(TestFunction):
    def __init__(self):
        TestFunction.__init__(self,
                              global_optimum=100,
                              fixed_accuracy=100,
                              x_optimum=np.ones((100), dtype=np.byte))

    def f(self, x):
        return np.sum(x, axis=1)


class Sphere(TestFunction):
    def __init__(self):
        TestFunction.__init__(self,
                              global_optimum=0,
                              fixed_accuracy=1e-6,
                              x_optimum=np.zeros((100), dtype=np.float64))

    def f(self, x):
        return np.sum(x**2, axis=-1)


class Schwefe1_2(TestFunction):
    def __init__(self):
        TestFunction.__init__(self,
                              global_optimum=0,
                              fixed_accuracy=1e-6,
                              x_optimum=np.zeros((100), dtype=np.float64))

    def f(self, x):
        return np.sum(np.add.accumulate(x, axis=-1)**2, axis=-1)


class HighConditionedElliptic(TestFunction):
    def __init__(self):
        TestFunction.__init__(self, global_optimum=0,
                              fixed_accuracy=1e-6,
                              x_optimum=np.zeros((100), dtype=np.float64))

    def f(self, x):
        i = np.arange(1, x.shape[1]+1)
        demension = x.shape[1]
        return np.sum((1e6**((i - 1)/(demension - 1)))*x**2, axis=-1)


class Rosenbrock(TestFunction):
    def __init__(self):
        TestFunction.__init__(self,
                              global_optimum=0,
                              fixed_accuracy=1e-6,
                              x_optimum=np.ones((100), dtype=np.float64))

    def f(self, x):
        value = 100*((x.T[:-1]**2 - x.T[1:])**2) + (x.T[:-1] - 1)**2
        return np.sum(value.T, axis=-1)


class Rastrigin(TestFunction):
    def __init__(self):
        TestFunction.__init__(self,
                              global_optimum=0,
                              fixed_accuracy=1e-6,
                              x_optimum=np.zeros((100), dtype=np.float64))

    def f(self, x):
        return np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10, axis=-1)


class Griewank(TestFunction):
    def __init__(self):
        TestFunction.__init__(self,
                              global_optimum=0,
                              fixed_accuracy=1e-6,
                              x_optimum=np.zeros((100), dtype=np.float64))

    def f(self, x):
        sqrt_i = np.sqrt(np.arange(1, x.shape[1]+1))
        sum_ = np.sum((x**2)/4000, axis=-1)
        prod_ = np.prod(np.cos(x/sqrt_i), axis=-1)
        return sum_ - prod_ + 1


class Ackley(TestFunction):
    def __init__(self):
        TestFunction.__init__(self,
                              global_optimum=0,
                              fixed_accuracy=1e-6,
                              x_optimum=np.zeros((100), dtype=np.float64))

    def f(self, x):
        a = 20
        c = 2*np.pi
        b = 0.2
        D = x.shape[1]
        left = -a*np.exp(-b*np.sqrt(np.sum(x**2, axis=1)/D))
        right = -np.exp((1/D)*np.sum(np.cos(c*x), axis=1))
        return left + right + a + np.exp(1)


class Weierstrass(TestFunction):
    def __init__(self):
        TestFunction.__init__(self,
                              global_optimum=0,
                              fixed_accuracy=1e-6,
                              x_optimum=np.zeros((100), dtype=np.float64))

    def f(self, x):
        a = 0.5
        b = 3
        k_max = np.arange(20, dtype=np.int64)
        D = x.shape[1]

        two_pi_power_b = 2*np.pi*(b**k_max)
        a_power_k = a**k_max
        arg1 = two_pi_power_b*(x[:, :, np.newaxis]+0.5)
        arg2 = two_pi_power_b*0.5
        left = a_power_k*np.cos(arg1)
        right = a_power_k*np.cos(arg2)

        left_sum = np.sum(np.sum(left, axis=-1), axis=-1)
        right_sum = np.sum(right, axis=-1)
        return left_sum - D*right_sum


# CEC05 #1
class ShiftedSphere(TestShiftedFunction, Sphere):
    def __init__(self):
        Sphere.__init__(self)
        TestShiftedFunction.__init__(self,
                                     global_optimum=fbias_data[0],
                                     fixed_accuracy=1e-6,
                                     x_optimum=sphere_func_data)


# CEC05 #2
class ShiftedSchwefe1_2(TestShiftedFunction, Schwefe1_2):
    def __init__(self):
        Schwefe1_2.__init__(self)
        TestShiftedFunction.__init__(self,
                                     global_optimum=fbias_data[1],
                                     fixed_accuracy=1e-6,
                                     x_optimum=schwefel_102_data)


# CEC05 #3
class ShiftedRotatedHighConditionedElliptic(TestShiftedRotatedFunction,
                                            HighConditionedElliptic):
    def __init__(self):
        HighConditionedElliptic.__init__(self)
        TestShiftedRotatedFunction.__init__(
            self,
            global_optimum=fbias_data[2],
            fixed_accuracy=fbias_data[2] + 1e-6,
            x_optimum=high_cond_elliptic_rot_data,
            rotate_M_D2=elliptic_M_D2,
            rotate_M_D10=elliptic_M_D10,
            rotate_M_D30=elliptic_M_D30,
            rotate_M_D50=elliptic_M_D50)


# CEC05 #4
class ShiftedSchwefe1_2WithNoise(TestShiftedFunction, Schwefe1_2):
    def __init__(self):
        Schwefe1_2.__init__(self)
        TestShiftedFunction.__init__(self,
                                     global_optimum=fbias_data[3],
                                     fixed_accuracy=1e-6,
                                     x_optimum=schwefel_102_data)

    def f(self, x):
        value = np.sum(np.add.accumulate(x, axis=-1)**2, axis=-1)
        noise = np.abs(np.random.normal(size=value.shape))
        return value*(1 + 0.4*noise)


# CEC05 #5
class Schwefel2_6(TestFunction):
    def __init__(self):
        TestFunction.__init__(self,
                              global_optimum=fbias_data[4],
                              fixed_accuracy=1e-6,
                              x_optimum=o_206)

    def f(self, x):
        D = x.shape[1]
        arange = np.arange(self.x_optimum.shape[0])
        cond_1 = arange + 1 <= np.ceil(D / 4.0)
        cond_2 = arange + 1 >= np.floor((3.0 * D) / 4.0)
        self.x_optimum[cond_1] = -100
        self.x_optimum[cond_2] = 100

        o = self.x_optimum[:x.shape[1]]

        A = A_206[:x.shape[1], :x.shape[1]]
        Ax = A@x.T
        B = A@o
        fx = np.abs(Ax - B[:, np.newaxis])
        return np.max(fx, axis=0) + self.global_optimum


# CEC05 #6
class ShiftedRosenbrock(TestShiftedFunction, Rosenbrock):
    def __init__(self):
        Rosenbrock.__init__(self)
        TestShiftedFunction.__init__(self,
                                     global_optimum=fbias_data[5],
                                     fixed_accuracy=1e-2,
                                     x_optimum=rosenbrock_func_data)

    def shift(self, x):
        shape = x.shape
        axis = [1]*(len(shape)-1) + [-1]
        return x - self.x_optimum[:shape[-1]].reshape(axis) + 1


# CEC05 #7
class ShiftedRotatedGriewank(TestShiftedRotatedFunction,
                             Griewank):

    def __init__(self):
        Griewank.__init__(self)
        TestShiftedRotatedFunction.__init__(
            self,
            global_optimum=fbias_data[6],
            fixed_accuracy=1e-2,
            x_optimum=griewank_func_data,
            rotate_M_D2=griewank_M_D2,
            rotate_M_D10=griewank_M_D10,
            rotate_M_D30=griewank_M_D30,
            rotate_M_D50=griewank_M_D50)


# CEC05 #8
class ShiftedRotatedAckley(TestShiftedRotatedFunction,
                           Ackley):
    def __init__(self):
        Ackley.__init__(self)
        TestShiftedRotatedFunction.__init__(
            self,
            global_optimum=fbias_data[7],
            fixed_accuracy=1e-2,
            x_optimum=ackley_func_data,
            rotate_M_D2=ackley_M_D2,
            rotate_M_D10=ackley_M_D10,
            rotate_M_D30=ackley_M_D30,
            rotate_M_D50=ackley_M_D50)

    def shift(self, x):
        shape = x.shape
        self.x_optimum[:shape[-1]][::2] = -32.0

        axis = [1]*(len(shape)-1) + [-1]
        return x - self.x_optimum[:shape[-1]].reshape(axis)


# CEC05 #9
class ShiftedRastrigin(TestShiftedFunction, Rastrigin):
    def __init__(self):
        Rastrigin.__init__(self)
        TestShiftedFunction.__init__(self,
                                     global_optimum=fbias_data[8],
                                     fixed_accuracy=1e-2,
                                     x_optimum=rastrigin_func_data)


# CEC05 #10
class ShiftedRotatedRastrigin(TestShiftedRotatedFunction,
                              Rastrigin):
    def __init__(self):
        Rastrigin.__init__(self)
        TestShiftedRotatedFunction.__init__(
            self,
            global_optimum=fbias_data[9],
            fixed_accuracy=1e-2,
            x_optimum=rastrigin_func_data,
            rotate_M_D2=rastrigin_M_D2,
            rotate_M_D10=rastrigin_M_D10,
            rotate_M_D30=rastrigin_M_D30,
            rotate_M_D50=rastrigin_M_D50)


# CEC05 #11
class ShiftedRotatedWeierstrass(TestShiftedRotatedFunction,
                                Weierstrass):
    def __init__(self):
        Weierstrass.__init__(self)
        TestShiftedRotatedFunction.__init__(
            self,
            global_optimum=fbias_data[10],
            fixed_accuracy=1e-2,
            x_optimum=weierstrass_data,
            rotate_M_D2=weierstrass_M_D2,
            rotate_M_D10=weierstrass_M_D10,
            rotate_M_D30=weierstrass_M_D30,
            rotate_M_D50=weierstrass_M_D50)


# CEC05 #12
class Schwefel2_13(TestFunction):
    def __init__(self):
        TestFunction.__init__(self,
                              global_optimum=fbias_data[11],
                              fixed_accuracy=1e-2,
                              x_optimum=alpha_213)

    def f(self, x):
        a = a_213[:x.shape[1], :x.shape[1]]
        b = b_213[:x.shape[1], :x.shape[1]]
        alpha = alpha_213[:x.shape[1]]

        A = np.sum(a*np.sin(alpha)[np.newaxis:,]
                   + b*np.cos(alpha)[np.newaxis:,], axis=1)

        B1 = a[np.newaxis]*np.sin(x[:, np.newaxis])
        B2 = b[np.newaxis]*np.cos(x[:, np.newaxis])
        B = np.sum(B1 + B2, axis=-1)

        fx = np.sum((A[np.newaxis:,] - B)**2, axis=-1)

        return fx + self.global_optimum
