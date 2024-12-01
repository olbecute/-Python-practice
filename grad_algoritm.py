import time
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(x) + 0.5*x

def df(x):
    return np.cos(x) + 0.5

N = 20                              # число итераций
xx = 2.5                              # начальное значение
lmd = 0.3                           # шаг сходимости

x_plt = np.arange(-5.0, 5.0, 0.1)      # от -5 до 5 с шагом 0.1
f_plt = [f(x) for x in x_plt]       # визуализация функции

plt.ion()                           # включение интерактивного режима отображения графиков (перемещения точки)
fig, ax = plt.subplots()            # создание окуна и осей для графика
ax.grid(True)                       # отображения сетки на графике

ax.plot(x_plt, f_plt)                       # отображение параболы
point = ax.scatter(xx, f(xx), c='red')      # отображение точки


mn = 100
# цикл для N итераций:
for i in range(N):
    lmb = 1 / min(i + 1, mn)    
    xx = xx - lmd*np.sign(df(xx))            # изменение аргумента на текущей итерации
    
    point.set_offsets([xx, f(xx)])  # отображение нового положения точки
    
    # перерисовка графика с задержкой 0.02 мс
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.02)
    
# отключение интерактивного режима отображения графика
plt.ioff()
print(xx)
ax.scatter(xx, f(xx), c='blue')
plt.show()
