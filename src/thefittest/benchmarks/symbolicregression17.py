import numpy as np


def z(x):
    first = - 1/((x - 1)**2 + 0.2)
    left = - 1/(2*(x - 2)**2 + 0.15)
    right = - 1/(3*(x - 3)**2 + 0.3)
    return first + left + right


def F1(x):
    firts = 0.05*(x[:, 0]-1)*(x[:, 0]-1)
    exp_x_power_2 = np.exp(-2.77257*x[:, 0]*x[:, 0])
    left = (3 - 2.9*exp_x_power_2)
    right = (1 - np.cos(x[:, 0]*(4-50*exp_x_power_2)))
    return firts + left*right


def F2(x):
    left = 0.5*np.cos(1.5*(10*x[:, 0]-0.3))*np.cos(31.4*x[:, 0])
    right = 0.5*np.cos(np.sqrt(5)*10*x[:, 0])*np.cos(35*x[:, 0])
    return 1 - left + right


def F3(x):
    left = 0.1*x[:, 0]**2 + 0.1*x[:, 1]**2
    right = - 4*np.cos(0.8*x[:, 0]) - 4*np.cos(0.8*x[:, 1]) + 8
    return left + right


def F4(x):
    left = (0.1*1.5*x[:, 1])**2 + (0.1*0.8*x[:, 0])**2
    right = -4*np.cos(0.8*1.5*x[:, 1]) - 4*np.cos(0.8*0.8*x[:, 0]) + 8
    return left + right


def F5(x):
    return 100*((x[:, 1]-x[:, 0]**2))**2 + (1 - x[:, 0])**2


def F6(x):
    left = 0.005*(x[:, 0]**2 + x[:, 1]**2)
    right = -np.cos(x[:, 0])*np.cos(x[:, 1]/np.sqrt(2)) + 2
    return -10/(left + right) + 10


def F7(x):
    down = 100*(x[:, 0]**2 - x[:, 1]) + (1 - x[:, 0])**2 + 1
    return -100/down + 100


def F8(x):
    power_x_y = x[:, 0]**2 + x[:, 1]**2
    up = 1-np.sin(np.sqrt(power_x_y))**2
    down = 1 + 0.001*(power_x_y)
    return up/down


def F9(x):
    first = 0.5*(x[:, 0]**2 + x[:, 1]**2)
    left = 2*0.8 + 0.8*np.cos(1.5*x[:, 0])*np.cos(3.14*x[:, 1])
    right = 0.8*np.cos(np.sqrt(5)*x[:, 0])*np.cos(3.5*x[:, 1])
    return first*(left + right)


def F10(x):
    first = 0.5*(x[:, 0]**2 + x[:, 1]**2)
    left = 2*0.8 + 0.8*np.cos(1.5*x[:, 0])*np.cos(3.14*x[:, 1])
    right = 0.8*np.cos(np.sqrt(5)*x[:, 0])*np.cos(3.5*x[:, 1])
    return first*(left + right)


def F11(x):
    left = (x[:, 0]**2)*np.abs(np.sin(2*x[:, 0]))
    right = (x[:, 1]**2)*np.abs(np.sin(2*x[:, 1]))
    last = -1/(5*x[:, 0]**2 + 5*x[:, 1]**2 + 0.2) + 5
    return left + right + last


def F12(x):
    first = 0.5*(x[:, 0]**2 + x[:, 0]*x[:, 1] + x[:, 1]**2)
    left = 1 + 0.5*np.cos(1.5*x[:, 0])*np.cos(3.2 *
                                              x[:, 0]*x[:, 1])*np.cos(3.14*x[:, 1])
    right = 0.5*np.cos(2.2*x[:, 1])*np.cos(4.8*x[:, 0]
                                           * x[:, 1])*np.cos(3.5*x[:, 1])
    return first*(left + right)


def F13(x):
    z_1 = z(x[:, 0])
    z_2 = z(x[:, 1])
    return -z_1*z_2


def F14(x):
    z_1 = z(x[:, 0])
    z_2 = z(x[:, 1])
    return z_1 + z_2


def F15(x):
    return (x[:, 0] - 2)**2 + (x[:, 1] - 1)**2


def F16(x):
    return np.sin(x[:, 0])*x[:, 0]*x[:, 0]


def F17(x):
    return np.sin(x[:, 0]) + x[:, 0]


problems_dict = {'F1': {'function': F1, 'bounds': (-1, 1), 'n_vars': 1},
                 'F2': {'function': F2, 'bounds': (-1, 1), 'n_vars': 1},
                 'F3': {'function': F3, 'bounds': (-16, 16), 'n_vars': 2},
                 'F4': {'function': F4, 'bounds': (-16, 16), 'n_vars': 2},
                 'F5': {'function': F5, 'bounds': (-2, 2), 'n_vars': 2},
                 'F6': {'function': F6, 'bounds': (-16, 16), 'n_vars': 2},
                 'F7': {'function': F7, 'bounds': (-5, 5), 'n_vars': 2},
                 'F8': {'function': F8, 'bounds': (-10, 10), 'n_vars': 2},
                 'F9': {'function': F9, 'bounds': (-2.5, 2.5), 'n_vars': 2},
                 'F10': {'function': F10, 'bounds': (-5, 5), 'n_vars': 2},
                 'F11': {'function': F11, 'bounds': (-4, 4), 'n_vars': 2},
                 'F12': {'function': F12, 'bounds': (0, 4), 'n_vars': 2},
                 'F13': {'function': F13, 'bounds': (0, 4), 'n_vars': 2},
                 'F14': {'function': F14, 'bounds': (0, 4), 'n_vars': 2},
                 'F15': {'function': F15, 'bounds': (-5, 5), 'n_vars': 2},
                 'F16': {'function': F16, 'bounds': (-5, 5), 'n_vars': 1},
                 'F17': {'function': F17, 'bounds': (-5, 5), 'n_vars': 1}}
