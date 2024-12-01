import numpy as np 
import matplotlib.pyplot as plt 

# Сигмоидная функция потерь
def loss(w, x, y):
    M = np.dot(w, x) * y
    return 2 / (1 + np.exp(M))

# Производная сигмоидной функции потерь по вертору w
def df(w, x, y):
    M = np.dot(w, x) * y
    return -2 * (1 + np.exp(M)) ** (-2) * np.exp(M) * x * y

# Обучающая выборка с тремя признаками (где третий признак - константа +1)
x_train = [
    [10, 50], [20, 30],
    [25, 30], [20, 60],
    [15, 70], [40, 40],
    [30, 45], [20, 45],
    [40, 30], [7, 35]
]
x_train = [x + [1] for x in x_train]    # Добавляем третий признак
x_train = np.array(x_train)
y_train = np.array([-1, 1, 1, -1, -1, 1, 1, -1, 1, -1])

n_train = len(x_train)   # Размер обучающей выборки
w = [0.0, 0.0, 0.0]     # начальные высовые коэффициенты
nt = 0.0005     # шаг сходимости SGD
lm = 0.01       # скорость забывания Q
N = 500         # число итераций

Q = np.mean([loss(x, w, y) for x, y in zip(x_train, y_train)])   # Показатель качества
Q_plot = [Q] # Формирование графика изменения Q в зависимости от работы алгоритма

for i in range(N):
    k = np.random.randint(0, n_train - 1) # Случайный индекс
    ek = loss(w, x_train[k], y_train[k])  # Вычисление потерь для выбранного k вектора
    w = w - nt * df(w, x_train[k], y_train[k]) # Корректировка весов по SGD
    Q = lm * ek + (1 - lm) * Q # пересчет показателя качества
    Q_plot.append(Q)
    
    
print(w)
print(Q_plot)

line_x = list(range(max(x_train[:, 0]))) # формирование графика разделяющей линии
line_y = [-x * w[0] / w[1] - w[2] for x in line_x] 

x_0 = x_train[y_train == 1] #  Формирование точек для 1ого класса
x_1 = x_train[y_train == -1] #  Формирование точек для 2ого класса

plt.scatter(x_0[:, 0], x_0[:, 1], color='red')
plt.scatter(x_1[:, 0], x_1[:, 1], color='blue')
plt.plot(line_x, line_y, color='green')

plt.xlim([0, 45])
plt.ylim([0, 75])
plt.ylabel(['длина'])
plt.xlabel(['ширина'])
plt.grid(True)
plt.show()