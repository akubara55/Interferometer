# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import nnoptic as nn
from decimal import Decimal, getcontext
import os, os.path


getcontext().prec = 3


@np.vectorize
def float2decimal(x):
    return Decimal(x)


@np.vectorize
def decimal2float(x):
    return float(x)


def create_alpha_coeff(r, n):
    alpha_coeff_r = np.zeros((n, n))
    for i in range(1, n):
        c = 1
        number_of_channels = np.arange(n, dtype='float64')
        number_of_channels[0] = 1/r
        number_r_0 = number_of_channels*r
        alpha_coeff_r[0] = number_r_0
        for j in range(n):
            if j < i:
                alpha_coeff_r[i, j] = number_r_0[i-j]
            if j > i:
                alpha_coeff_r[i, j] = number_r_0[c]
                c += 1
    alpha_coeff_r[0, 0] -= 1
    return float2decimal(alpha_coeff_r)


def create_alpha(r, n, case, function, q_array = 0, q = 0):
    alpha_coeff = create_alpha_coeff(r, n)
    if case == '1':
        for i in range(n):
            for j in range(n):
                if function == "Экспонента":
                    alpha_coeff[i, j] = (-q*alpha_coeff[i, j]).exp()
                if function == "Степенная":
                    if alpha_coeff[i, j] != 0:
                        alpha_coeff[i, j] = q/(alpha_coeff[i, j])
                    else:
                        alpha_coeff[i, j] = 1
    if case == '2':
        for i in range(n):
            for j in range(n):
                if function == "Экспонента":
                    alpha_coeff[i, j] = (-alpha_coeff[i, j]*q_array[j]).exp()
                if function == "Степенная":
                    if alpha_coeff[i, j] != 0:
                        alpha_coeff[i, j] = q_array/(alpha_coeff[i, j])
                    else:
                        alpha_coeff[i, j] = 1
    return decimal2float(alpha_coeff)


if __name__ == '__main__':
    fig, axes = plt.subplots(nrows=2, ncols=1)
    folder = 'модельные предположения'
    # direc = os.mkdir(folder)
    # Константы и рассматриваемый случай
    n = 4
    r = float(input("Введите расстояние между фазовращателями: "))
    coeff = float(input("Введите коэффициент ошибки для перемешивающих матриц: "))
    number = int(input("Калибруется фазовращатель номер "))
    function = input("Какой вид зависимости? ")
    # Создание перемешивающих матриц
    u_1 = np.loadtxt('первая_u.csv', dtype='complex')
    u_2 = np.loadtxt('вторая_u.csv', dtype='complex')
    list_u_l = [u_1, u_2]
    list_noisy_ul = nn.get_list_noisy(list_u_l, coeff, n)
    # Создание матрицы силы тока
    I_max = 19
    step_I = 0.015
    I_coeff = np.loadtxt('cила_тока.csv')
    I = step_I*I_coeff
    # Создание начальных данных
    a_in = np.loadtxt('излучение.csv', dtype='complex')
    alpha_0 = np.loadtxt('альфа.csv', dtype='float64')
    f_0 = np.loadtxt('нач_фаза.csv', dtype='float64')
    case = input("Коэффициент гамма - постоянная или характеристика элемента? ")
    if case == '1':
        q_float = float(input("Введите коэффицент гамма: "))
        ans = input("Учитываем воздействие? ")
        if ans == 'Да':
            name = f'Ф - {number}, \N{greek small letter gamma} = {q_float}, большое кросс-воздействие'
        if ans == 'Нет':
            name = f'Ф - {number}, \N{greek small letter gamma} = {q_float}, маленькое кросс-воздействие'
        axes[0].set(title=name)
        q_d = Decimal(q_float)
        # Создание матрицы влияния
        alpha_coeff = create_alpha(r, n, case, function, q=q_d)
    if case == '2':
        q_array = np.array([])
        for count in range(1, 5):
            q_i = float(input(f"Коэффициент гамма для {count} фазовращателя: "))
            q_array = np.append(q_array, q_i)
        q_array = float2decimal(q_array)
        name = f"Ф - {number}, \N{greek small letter gamma} - разные"
        axes[0].set(title=name)
        # Создание матрицы влияния
        alpha_coeff = create_alpha(r, n, case, function, q_array=q_array)
    alpha_01 = alpha_0[0]
    alpha_02 = alpha_0[1]
    alpha_03 = alpha_0[2]
    alpha_1 = alpha_coeff*alpha_01.T
    alpha_2 = alpha_coeff*alpha_02.T
    alpha_3 = alpha_coeff*alpha_03.T
    a1, a2, a3, a4 = [], [], [], []
    I_list = []
    f_list1, f_list2, f_list3, f_list4 = [], [], [], []
    a_list = [a1, a2, a3, a4]
    f_list = [f_list1, f_list2, f_list3, f_list4]
    for k in range(0, int(I_max//step_I)):
        I[number - 1, 1] = step_I*k
        I_list.append(I[number-1, 1]**2)
        f_1 = f_0[:, 0] + alpha_1.dot(I[:, 0]**2)
        f_2 = f_0[:, 1] + alpha_2.dot(I[:, 1]**2)
        f_3 = f_0[:, 2] + alpha_3.dot(I[:, 2]**2)
        f0 = np.vstack([f_1, f_2, f_3])
        list_fl = nn.create_list_fl(f0)
        _u = nn.interferometer(list_fl, list_noisy_ul, n)
        a_out = _u.dot(a_in)
        for p in range(4):
            a_list[p].append(abs(a_out[p])**2)
            f_list[p].append(f0[1, p])
    for i in range(n):
        axes[0].plot(I_list, a_list[i], label=f"К {i+1}")
        axes[1].plot(I_list, f_list[i], label=f"Ф 2.{i+1}")
        for j in range(2):
            axes[j].grid(which='major',
                         color='k')
            axes[j].minorticks_on()
            axes[j].grid(which='minor',
                         color='gray',
                         linestyle=':')
            axes[j].legend(loc='upper left', fontsize=8)
            axes[j].set_xlabel('Квадрат силы тока', fontsize=14)
    axes[0].set_ylabel('Мощность', fontsize=14)
    axes[1].set_ylabel('Сдвиг фаз', fontsize=14)
    graph = f"{case} {name} {function}.png"
    plt.savefig(os.path.join(folder, graph))
    plt.show()
    # Тест - сравнение с экспериментально полученными данными
    fig, axes = plt.subplots(nrows=2, ncols=1)
    alpha_test = np.loadtxt('альфа_эксп.csv')
    I_test = np.loadtxt('I_эксп.csv').reshape(4, 1)
    u1 = np.loadtxt('u_1_эксп.csv', dtype='complex64')
    u2 = np.loadtxt('u_2_эксп.csv', dtype='complex64')
    a_init_test = np.loadtxt('a_init_эксп.csv', dtype='complex64').reshape((4, 1))
    q = Decimal('1.98244711')
    alpha_coeff = create_alpha(r, n, '1', 'Экспонента', q=q)
    alpha = alpha_coeff*alpha_test
    a1, a2, a3, a4 = [], [], [], []
    I_list = []
    f_list1, f_list2, f_list3, f_list4 = [], [], [], []
    a_list = [a1, a2, a3, a4]
    f_list = [f_list1, f_list2, f_list3, f_list4]
    a_out_exp = np.loadtxt('ch13, 1.txt')[0:, 1:]
    for i in range(0, 1310, 10):
        I_test[0, 0] = i*0.015
        I_list.append(I_test[0, 0]**2)
        f2 = alpha.dot(I_test**2)
        #f_1, f_3 = np.zeros(f_2.shape), np.zeros(f_2.shape)
        # f0 = np.vstack([f_1, f_2, f_3])
        # print(f0)
        list_fl_ = nn.create_list_fl(f2.T)
        #_u = nn.interferometer(list_fl, list_u, n)
        _u = u2.dot(list_fl_[0].dot(u1))
        a_out = _u.dot(a_init_test)
        for p in range(4):
            a_list[p].append(abs(a_out[p, 0])**2)
            f_list[p].append(f2[p, 0])
    color_sc = ['b', 'r', 'y', 'm']
    color_pl = ['r', 'b', 'm', 'y']
    for i in range(n):
        axes[0].plot(I_list, a_list[i], label=f"К {i+1}", color=color_pl[i])
        axes[0].scatter(I_list, a_out_exp[:, i], color=color_sc[i])
        axes[1].plot(I_list, f_list[i], label=f"Ф 2.{i+1}")
        for j in range(2):
            axes[j].grid(which='major',
                         color='k')
            axes[j].minorticks_on()
            axes[j].grid(which='minor',
                         color='gray',
                         linestyle=':')
            axes[j].legend(loc='upper left', fontsize=8)
            axes[j].set_xlabel('Квадрат силы тока', fontsize=14)
    axes[0].set_ylabel('Мощность', fontsize=14)
    axes[1].set_ylabel('Сдвиг фаз', fontsize=14)
    plt.show()