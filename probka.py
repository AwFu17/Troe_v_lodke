import numpy as np
import matplotlib.pyplot as plt
import math
from enum import Enum
from typing import List, Tuple, Callable

# ==================== ЧАСТЬ 1: ОПРЕДЕЛЕНИЕ КЛАССОВ ====================

class BodyType(Enum):
    """Перечисление для типов тел"""
    CIRCLE = "circle"
    SQUARE = "square"

class MaterialPoint:
    """
    Класс материальной точки.
    Хранит текущие координаты и историю движения.
    """
    
    def __init__(self, x: float, y: float, point_id: int = 0):
        # Инициализация точки начальными координатами
        self.x = x  # Текущая координата X
        self.y = y  # Текущая координата Y
        self.point_id = point_id  # Уникальный идентификатор точки
        
        # Списки для хранения всей траектории движения
        self.trajectory_x = [x]  # Начинаем с начальной позиции
        self.trajectory_y = [y]  # Начинаем с начальной позиции
    
    def update_position(self, new_x: float, new_y: float):
        """
        Обновление координат точки и сохранение в историю.
        
        Args:
            new_x: Новая координата X
            new_y: Новая координата Y
        """
        self.x = new_x
        self.y = new_y
        # Добавляем новые координаты в историю
        self.trajectory_x.append(new_x)
        self.trajectory_y.append(new_y)

class Body:
    """
    Класс деформируемого тела.
    Состоит из множества материальных точек.
    """
    
    def __init__(self, body_type: BodyType, size: float, quarter: int = 2):
        """
        Инициализация тела.
        
        Args:
            body_type: Тип тела (CIRCLE или SQUARE)
            size: Размер (радиус для круга, сторона для квадрата)
            quarter: Координатная четверть (1-4)
        """
        self.type = body_type  # Тип тела
        self.size = size      # Размер тела
        self.quarter = quarter  # Четверть расположения
        self.points = []      # Список материальных точек
        self.points_per_side = 0  # Количество точек на стороне (для квадрата)
        
        # Генерируем точки тела при создании
        self._generate_points()
    
    def _generate_points(self, num_points: int = 100):
        """
        Генерация материальных точек для тела.
        
        Args:
            num_points: Количество точек для генерации
        """
        self.points = []  # Очищаем список точек
        point_id = 0      # Счетчик идентификаторов
        
        if self.type == BodyType.CIRCLE:
            self._generate_circle_points(num_points, point_id)
        else:  # BodyType.SQUARE
            self._generate_square_points(num_points, point_id)
    
    def _generate_circle_points(self, num_points: int, start_id: int):
        """
        Генерация точек для круга.
        Круг располагается так, чтобы не касаться осей координат.
        """
        # Для 2-й четверти: x < 0, y > 0
        # Центр круга смещаем от осей на 0.5
        center_x = -self.size - 0.5
        center_y = self.size + 0.5
        
        # Создаем равномерно распределенные углы от 0 до 2π
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        
        # Для каждого угла создаем точку на окружности
        for i, angle in enumerate(angles):
            # Параметрическое уравнение окружности
            x = center_x + self.size * np.cos(angle)
            y = center_y + self.size * np.sin(angle)
            
            # Создаем материальную точку
            point = MaterialPoint(x, y, start_id + i)
            self.points.append(point)
    
    def _generate_square_points(self, num_points: int, start_id: int):
        """
        Генерация точек для квадрата.
        Точки располагаются по периметру квадрата.
        """
        # Для 2-й четверти: левый нижний угол (-size-0.5, 0.5)
        x0 = -self.size - 0.5  # Левая граница
        y0 = 0.5               # Нижняя граница
        
        # Количество точек на каждой стороне
        self.points_per_side = num_points // 4
        point_id = start_id
        
        # Генерация точек для каждой из 4 сторон
        
        # 1. Нижняя сторона (движение слева направо)
        for i in range(self.points_per_side):
            # Линейная интерполяция от x0 до x0+size
            x = x0 + (self.size * i / self.points_per_side)
            y = y0  # Постоянная Y
            self.points.append(MaterialPoint(x, y, point_id))
            point_id += 1
        
        # 2. Правая сторона (движение снизу вверх)
        for i in range(self.points_per_side):
            x = x0 + self.size  # Постоянная X
            y = y0 + (self.size * i / self.points_per_side)
            self.points.append(MaterialPoint(x, y, point_id))
            point_id += 1
        
        # 3. Верхняя сторона (движение справа налево)
        for i in range(self.points_per_side):
            x = x0 + self.size - (self.size * i / self.points_per_side)
            y = y0 + self.size  # Постоянная Y
            self.points.append(MaterialPoint(x, y, point_id))
            point_id += 1
        
        # 4. Левая сторона (движение сверху вниз)
        for i in range(self.points_per_side):
            x = x0  # Постоянная X
            y = y0 + self.size - (self.size * i / self.points_per_side)
            self.points.append(MaterialPoint(x, y, point_id))
            point_id += 1
    
    def get_initial_contour(self) -> Tuple[List[float], List[float]]:
        """
        Получение контура начального тела.
        
        Returns:
            Кортеж (x_coords, y_coords) - координаты контура
        """
        if self.type == BodyType.CIRCLE:
            # Для круга - больше точек для гладкого контура
            angles = np.linspace(0, 2 * np.pi, 100)
            center_x = -self.size - 0.5
            center_y = self.size + 0.5
            
            x_coords = [center_x + self.size * np.cos(a) for a in angles]
            y_coords = [center_y + self.size * np.sin(a) for a in angles]
            
        else:  # SQUARE
            # Для квадрата - 5 точек (4 угла + первая для замыкания)
            x0 = -self.size - 0.5
            y0 = 0.5
            
            x_coords = [x0, x0 + self.size, x0 + self.size, x0, x0]
            y_coords = [y0, y0, y0 + self.size, y0 + self.size, y0]
        
        return x_coords, y_coords
    
    def get_deformed_contour(self, time_index: int = -1) -> Tuple[List[float], List[float]]:
        """
        Получение контура деформированного тела с сохранением порядка точек.
        
        Args:
            time_index: Индекс момента времени (-1 - последний)
            
        Returns:
            Кортеж (x_coords, y_coords) - координаты контура
        """
        x_coords = []
        y_coords = []
        
        if self.type == BodyType.CIRCLE:
            # ДЛЯ КРУГА: берем ВСЕ точки в исходном порядке
            for point in self.points:
                if len(point.trajectory_x) > abs(time_index):
                    idx = time_index if time_index >= 0 else len(point.trajectory_x) + time_index
                    idx = max(0, min(idx, len(point.trajectory_x) - 1))
                    x_coords.append(point.trajectory_x[idx])
                    y_coords.append(point.trajectory_y[idx])
            
            # Замыкаем окружность (соединяем последнюю точку с первой)
            if len(x_coords) > 0:
                x_coords.append(x_coords[0])
                y_coords.append(y_coords[0])
                
        else:  # SQUARE
            # ДЛЯ КВАДРАТА: берем точки в правильном порядке по периметру
            if self.points_per_side == 0:
                # Если points_per_side не установлен, вычисляем его
                self.points_per_side = len(self.points) // 4
            
            # 1. Нижняя сторона (индексы 0 до points_per_side-1)
            for i in range(self.points_per_side):
                point = self.points[i]
                if len(point.trajectory_x) > abs(time_index):
                    idx = time_index if time_index >= 0 else len(point.trajectory_x) + time_index
                    idx = max(0, min(idx, len(point.trajectory_x) - 1))
                    x_coords.append(point.trajectory_x[idx])
                    y_coords.append(point.trajectory_y[idx])
            
            # 2. Правая сторона (индексы points_per_side до 2*points_per_side-1)
            for i in range(self.points_per_side, 2 * self.points_per_side):
                point = self.points[i]
                if len(point.trajectory_x) > abs(time_index):
                    idx = time_index if time_index >= 0 else len(point.trajectory_x) + time_index
                    idx = max(0, min(idx, len(point.trajectory_x) - 1))
                    x_coords.append(point.trajectory_x[idx])
                    y_coords.append(point.trajectory_y[idx])
            
            # 3. Верхняя сторона (индексы 2*points_per_side до 3*points_per_side-1)
            for i in range(2 * self.points_per_side, 3 * self.points_per_side):
                point = self.points[i]
                if len(point.trajectory_x) > abs(time_index):
                    idx = time_index if time_index >= 0 else len(point.trajectory_x) + time_index
                    idx = max(0, min(idx, len(point.trajectory_x) - 1))
                    x_coords.append(point.trajectory_x[idx])
                    y_coords.append(point.trajectory_y[idx])
            
            # 4. Левая сторона (индексы 3*points_per_side до 4*points_per_side-1)
            for i in range(3 * self.points_per_side, 4 * self.points_per_side):
                point = self.points[i]
                if len(point.trajectory_x) > abs(time_index):
                    idx = time_index if time_index >= 0 else len(point.trajectory_x) + time_index
                    idx = max(0, min(idx, len(point.trajectory_x) - 1))
                    x_coords.append(point.trajectory_x[idx])
                    y_coords.append(point.trajectory_y[idx])
            
            # Замыкаем квадрат (возвращаемся к первой точке)
            if len(x_coords) > 0:
                x_coords.append(x_coords[0])
                y_coords.append(y_coords[0])
        
        return x_coords, y_coords
    
    def get_intermediate_contours(self, num_contours: int = 3) -> List[Tuple[List[float], List[float]]]:
        """
        Получение нескольких промежуточных контуров для анимации.
        
        Args:
            num_contours: Количество промежуточных контуров
            
        Returns:
            Список контуров [(x1, y1), (x2, y2), ...]
        """
        contours = []
        
        # Равномерно распределяем моменты времени
        total_steps = len(self.points[0].trajectory_x) if self.points else 1
        step_indices = np.linspace(0, total_steps - 1, num_contours + 2, dtype=int)[1:-1]
        
        for idx in step_indices:
            contour = self.get_deformed_contour(idx)
            contours.append(contour)
        
        return contours

class VelocityField:
    """
    Класс поля скоростей.
    Вычисляет скорость в любой точке пространства в любой момент времени.
    """
    
    def __init__(self, A_func: Callable[[float], float], 
                 B_func: Callable[[float], float]):
        """
        Инициализация поля скоростей.
        
        Args:
            A_func: Функция A(t) из уравнения v1 = -A(t)*x1
            B_func: Функция B(t) из уравнения v2 = B(t)*x2
        """
        self.A = A_func  # Функция для горизонтальной компоненты
        self.B = B_func  # Функция для вертикальной компоненты
    
    def get_velocity(self, x: float, y: float, t: float) -> Tuple[float, float]:
        """
        Вычисление вектора скорости в заданной точке.
        
        Args:
            x: Координата X точки
            y: Координата Y точки
            t: Время
            
        Returns:
            Кортеж (vx, vy) - компоненты вектора скорости
        """
        # Вычисляем компоненты скорости по формулам из условия
        vx = -self.A(t) * x  # v1 = -A(t)*x1
        vy = self.B(t) * y   # v2 = B(t)*x2
        
        return vx, vy

class ButcherTable:
    """
    Класс для хранения таблицы Бутчера.
    Таблица содержит коэффициенты метода Рунге-Кутты.
    """
    
    def __init__(self, c: np.ndarray, a: np.ndarray, b: np.ndarray):
        """
        Инициализация таблицы Бутчера.
        
        Args:
            c: Коэффициенты времени (узлы метода)
            a: Матрица коэффициентов для промежуточных вычислений
            b: Весовые коэффициенты для итогового результата
        """
        self.c = c  # Временные узлы (например, [0, 1] для начала и конца шага)
        self.a = a  # Матрица коэффициентов (s x s)
        self.b = b  # Весовые коэффициенты (например, [0.5, 0.5])

# Таблица Бутчера 2.2 из задания
BUTCHER_TABLE_2_2 = ButcherTable(
    c=np.array([0, 1]),                    # Два временных узла
    a=np.array([[0, 0], [1, 0]]),          # Матрица 2x2
    b=np.array([0.5, 0.5])                 # Веса 50/50
)

class RungeKuttaSolver:
    """
    Класс для численного интегрирования методом Рунге-Кутты.
    """
    
    def __init__(self, butcher_table: ButcherTable):
        """
        Инициализация решателя.
        
        Args:
            butcher_table: Таблица Бутчера с коэффициентами метода
        """
        self.table = butcher_table  # Сохраняем таблицу коэффициентов
    
    def solve_step(self, field: VelocityField, point: MaterialPoint, 
                   t: float, dt: float) -> Tuple[float, float]:
        """
        Выполнение одного шага интегрирования.
        
        Args:
            field: Поле скоростей для вычисления производных
            point: Текущее состояние материальной точки
            t: Текущее время
            dt: Шаг интегрирования
            
        Returns:
            Кортеж (new_x, new_y) - новые координаты точки
        """
        # Текущие координаты точки
        x, y = point.x, point.y
        
        # Количество стадий метода (равно длине массива c)
        num_stages = len(self.table.c)
        
        # Массивы для хранения вычисленных скоростей (k-коэффициентов)
        kx = np.zeros(num_stages)  # Для X-компоненты
        ky = np.zeros(num_stages)  # Для Y-компоненты
        
        # ЦИКЛ ПО СТАДИЯМ МЕТОДА
        for i in range(num_stages):
            # Вычисляем промежуточные координаты для i-й стадии
            sum_x = 0.0
            sum_y = 0.0
            
            # Суммируем вклады от предыдущих стадий
            for j in range(i):
                sum_x += self.table.a[i][j] * kx[j]
                sum_y += self.table.a[i][j] * ky[j]
            
            # Координаты для вычисления скорости на i-й стадии
            xi = x + dt * sum_x
            yi = y + dt * sum_y
            
            # Время для i-й стадии
            ti = t + dt * self.table.c[i]
            
            # ВЫЧИСЛЯЕМ СКОРОСТЬ в промежуточной точке
            kx[i], ky[i] = field.get_velocity(xi, yi, ti)
        
        # ВЫЧИСЛЯЕМ НОВЫЕ КООРДИНАТЫ с использованием весов b
        # new_x = x + dt * Σ(b[i] * kx[i])
        # new_y = y + dt * Σ(b[i] * ky[i])
        new_x = x + dt * np.dot(self.table.b, kx)
        new_y = y + dt * np.dot(self.table.b, ky)
        
        return new_x, new_y

class DeformationSimulator:
    """
    Главный класс симуляции деформации.
    Управляет всем процессом вычислений.
    """
    
    def __init__(self, body: Body, field: VelocityField, solver: RungeKuttaSolver):
        """
        Инициализация симулятора.
        
        Args:
            body: Деформируемое тело
            field: Поле скоростей
            solver: Численный метод интегрирования
        """
        self.body = body    # Тело для деформации
        self.field = field  # Поле скоростей
        self.solver = solver  # Метод интегрирования
        self.time_history = []  # История моментов времени
    
    def simulate(self, t_start: float, t_end: float, dt: float):
        """
        Запуск симуляции деформации.
        
        Args:
            t_start: Начальное время
            t_end: Конечное время
            dt: Шаг интегрирования
        """
        # Генерируем массив моментов времени
        self.time_history = np.arange(t_start, t_end + dt, dt)
        
        print(f"Запуск симуляции...")
        print(f"  Время: от {t_start} до {t_end}, шаг {dt}")
        print(f"  Количество шагов: {len(self.time_history)}")
        print(f"  Количество точек: {len(self.body.points)}")
        
        # ОСНОВНОЙ ЦИКЛ ИНТЕГРИРОВАНИЯ
        for step, t in enumerate(self.time_history):
            # Для каждой материальной точки тела
            for point in self.body.points:
                # Вычисляем новые координаты за шаг dt
                new_x, new_y = self.solver.solve_step(self.field, point, t, dt)
                
                # Обновляем положение точки
                point.update_position(new_x, new_y)
            
            # Вывод прогресса каждые 10% шагов
            if step % max(1, len(self.time_history) // 10) == 0:
                progress = (step / len(self.time_history)) * 100
                print(f"  Прогресс: {progress:.0f}%")
        
        print("Симуляция завершена!")

# ==================== ЧАСТЬ 2: ФУНКЦИИ ВИЗУАЛИЗАЦИИ ====================

def plot_velocity_field(field: VelocityField, t: float, 
                        x_lim: Tuple[float, float], 
                        y_lim: Tuple[float, float],
                        ax=None):
    """
    Построение поля скоростей в заданный момент времени.
    
    Args:
        field: Поле скоростей
        t: Момент времени
        x_lim: Границы по X
        y_lim: Границы по Y
        ax: Ось для рисования (если None, создается новая)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Создаем сетку точек
    x_vals = np.linspace(x_lim[0], x_lim[1], 15)
    y_vals = np.linspace(y_lim[0], y_lim[1], 15)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    
    # Вычисляем скорости в узлах сетки
    vx_grid = np.zeros_like(x_grid)
    vy_grid = np.zeros_like(y_grid)
    
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            vx, vy = field.get_velocity(x_grid[i, j], y_grid[i, j], t)
            vx_grid[i, j] = vx
            vy_grid[i, j] = vy
    
    # Нормируем векторы для наглядности
    magnitude = np.sqrt(vx_grid**2 + vy_grid**2)
    # Избегаем деления на ноль
    magnitude[magnitude == 0] = 1
    vx_norm = vx_grid / magnitude
    vy_norm = vy_grid / magnitude
    
    # Рисуем поле скоростей
    ax.quiver(x_grid, y_grid, vx_norm, vy_norm, magnitude,
              cmap='viridis', alpha=0.7, width=0.005, scale=30)
    
    ax.set_title(f"Поле скоростей при t = {t:.2f}", fontsize=12)
    ax.set_xlabel("Координата x₁")
    ax.set_ylabel("Координата x₂")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect('equal')

def plot_streamlines(field: VelocityField, t: float,
                     x_lim: Tuple[float, float],
                     y_lim: Tuple[float, float],
                     ax=None):
    """
    Построение линий тока в заданный момент времени.
    
    Args:
        field: Поле скоростей
        t: Момент времени
        x_lim: Границы по X
        y_lim: Границы по Y
        ax: Ось для рисования
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Более плотная сетка для линий тока
    x_vals = np.linspace(x_lim[0], x_lim[1], 30)
    y_vals = np.linspace(y_lim[0], y_lim[1], 30)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    
    # Вычисляем скорости
    vx_grid = np.zeros_like(x_grid)
    vy_grid = np.zeros_like(y_grid)
    
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            vx, vy = field.get_velocity(x_grid[i, j], y_grid[i, j], t)
            vx_grid[i, j] = vx
            vy_grid[i, j] = vy
    
    # Рисуем линии тока
    ax.streamplot(x_grid, y_grid, vx_grid, vy_grid,
                  color='purple', linewidth=1, density=2.0, arrowsize=1)
    
    ax.set_title(f"Линии тока при t = {t:.2f}", fontsize=12)
    ax.set_xlabel("Координата x₁")
    ax.set_ylabel("Координата x₂")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect('equal')

def plot_trajectories(body: Body, ax=None):
    """
    Построение траекторий движения материальных точек.
    
    Args:
        body: Деформируемое тело
        ax: Ось для рисования
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Рисуем траектории для каждой точки
    for i, point in enumerate(body.points):
        # Ограничиваем количество отображаемых траекторий для наглядности
        if i % max(1, len(body.points) // 50) == 0:
            ax.plot(point.trajectory_x, point.trajectory_y, 
                   'b-', alpha=0.2, linewidth=0.5)
    
    # Отмечаем начальные позиции
    init_x = [p.trajectory_x[0] for p in body.points]
    init_y = [p.trajectory_y[0] for p in body.points]
    ax.scatter(init_x, init_y, c='blue', s=10, alpha=0.5, label='Начало')
    
    # Отмечаем конечные позиции
    final_x = [p.trajectory_x[-1] for p in body.points]
    final_y = [p.trajectory_y[-1] for p in body.points]
    ax.scatter(final_x, final_y, c='red', s=10, alpha=0.5, label='Конец')
    
    ax.set_title("Траектории материальных точек", fontsize=12)
    ax.set_xlabel("Координата x₁")
    ax.set_ylabel("Координата x₂")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

def plot_body_deformation(body: Body, ax=None):
    """
    Построение начальной и деформированной формы тела.
    
    Args:
        body: Деформируемое тело
        ax: Ось для рисования
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Рисуем начальную форму
    init_x, init_y = body.get_initial_contour()
    ax.plot(init_x, init_y, 'b-', linewidth=2, label='Начальная форма')
    ax.fill(init_x, init_y, 'blue', alpha=0.2)
    
    # Рисуем деформированную форму
    deformed_x, deformed_y = body.get_deformed_contour(-1)
    ax.plot(deformed_x, deformed_y, 'r-', linewidth=2, label='Деформированная форма')
    
    # Для лучшей визуализации заполняем деформированную форму
    if len(deformed_x) > 0:
        ax.fill(deformed_x, deformed_y, 'red', alpha=0.2)
    
    # Рисуем промежуточные формы
    intermediate_contours = body.get_intermediate_contours(3)
    for i, (inter_x, inter_y) in enumerate(intermediate_contours):
        ax.plot(inter_x, inter_y, 'g--', linewidth=1, alpha=0.6,
                label=f'Промежуточная {i+1}' if i == 0 else "")
    
    ax.set_title(f"Деформация {body.type.value}", fontsize=12)
    ax.set_xlabel("Координата x₁")
    ax.set_ylabel("Координата x₂")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

# ==================== ЧАСТЬ 3: ОСНОВНАЯ ПРОГРАММА ====================

def main():
    """
    Главная функция программы.
    """
    print("=" * 60)
    print("СИМУЛЯТОР ДЕФОРМАЦИИ ТЕЛА")
    print("Подгруппа 1: Тарахова, Омаров, Соловьева")
    print("=" * 60)
    
    # ========== ШАГ 1: ВЫБОР ПАРАМЕТРОВ ПОЛЬЗОВАТЕЛЕМ ==========
    print("\nВыберите тип тела:")
    print("1. Круг радиуса 1")
    print("2. Квадрат со стороной 1")
    
    while True:
        choice = input("Ваш выбор (1 или 2): ").strip()
        if choice in ["1", "2"]:
            break
        print("Пожалуйста, введите 1 или 2")
    
    if choice == "1":
        body_type = BodyType.CIRCLE
        body_name = "круг радиуса 1"
    else:
        body_type = BodyType.SQUARE
        body_name = "квадрат со стороной 1"
    
    quarter = 2  # Для подгруппы 1 всегда 2-я четверть
    
    print(f"\nСоздаем тело: {body_name}")
    print(f"Расположение: {quarter}-я координатная четверть")
    
    # ========== ШАГ 2: СОЗДАНИЕ ОБЪЕКТОВ ==========
    
    # 1. Создаем тело
    body = Body(body_type, size=1.0, quarter=quarter)
    print(f"✓ Тело создано. Точек: {len(body.points)}")
    
    # 2. Определяем функции A(t) и B(t) из условия задачи
    #    Для подгруппы 1: A(t) = ln(t), B(t) = -e^t
    def A_func(t):
        """Функция A(t) = ln(t)"""
        # Используем math.log для натурального логарифма
        # Добавляем маленькое число для t близкого к 0
        if t <= 0:
            return math.log(1e-10)  # Очень маленькое значение
        return math.log(t)
    
    def B_func(t):
        """Функция B(t) = -e^t"""
        return -math.exp(t)
    
    # 3. Создаем поле скоростей
    velocity_field = VelocityField(A_func, B_func)
    print("✓ Поле скоростей создано")
    print("  Уравнения: v₁ = -ln(t)·x₁, v₂ = -e^t·x₂")
    
    # 4. Создаем решатель Рунге-Кутты
    solver = RungeKuttaSolver(BUTCHER_TABLE_2_2)
    print("✓ Решатель Рунге-Кутты создан")
    print(f"  Метод: {len(BUTCHER_TABLE_2_2.c)}-стадийный")
    
    # 5. Создаем симулятор
    simulator = DeformationSimulator(body, velocity_field, solver)
    print("✓ Симулятор деформации создан")
    
    # ========== ШАГ 3: НАСТРОЙКА ПАРАМЕТРОВ СИМУЛЯЦИИ ==========
    
    # Начинаем с t=1, так как ln(t) не определен при t=0
    t_start = 1.0
    t_end = 3.0    # Конечное время
    dt = 0.02      # Шаг интегрирования
    
    print(f"\nПараметры симуляции:")
    print(f"  Начальное время: t = {t_start}")
    print(f"  Конечное время: t = {t_end}")
    print(f"  Шаг интегрирования: dt = {dt}")
    
    # ========== ШАГ 4: ЗАПУСК СИМУЛЯЦИИ ==========
    print("\n" + "-" * 40)
    simulator.simulate(t_start, t_end, dt)
    print("-" * 40)
    
    # ========== ШАГ 5: ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ==========
    print("\nСтроим графики...")
    
    # Определяем границы для графиков
    # Собираем все координаты всех точек
    all_x = []
    all_y = []
    for point in body.points:
        all_x.extend(point.trajectory_x)
        all_y.extend(point.trajectory_y)
    
    # Вычисляем минимальные и максимальные значения
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    # Добавляем запас 20%
    x_margin = 0.2 * (x_max - x_min)
    y_margin = 0.2 * (y_max - y_min)
    
    x_lim = (x_min - x_margin, x_max + x_margin)
    y_lim = (y_min - y_margin, y_max + y_margin)
    
    # СОЗДАЕМ БОЛЬШОЙ ГРАФИК С 4 ПОДГРАФИКАМИ
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'Деформация {body_name} в поле скоростей', 
                fontsize=16, fontweight='bold')
    
    # График 1: Траектории движения
    plot_trajectories(body, axes[0, 0])
    
    # График 2: Деформация тела (Теперь правильно!)
    plot_body_deformation(body, axes[0, 1])
    
    # График 3: Поле скоростей в начальный момент
    plot_velocity_field(velocity_field, t_start, x_lim, y_lim, axes[1, 0])
    
    # График 4: Линии тока в конечный момент
    plot_streamlines(velocity_field, t_end, x_lim, y_lim, axes[1, 1])
    
    plt.tight_layout()
    
    # ДОПОЛНИТЕЛЬНЫЙ ГРАФИК: Сравнение форм в разные моменты времени
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    # Рисуем несколько контуров в разные моменты времени
    time_indices = [0, len(simulator.time_history)//4, 
                   len(simulator.time_history)//2,
                   len(simulator.time_history)-1]
    
    colors = ['blue', 'green', 'orange', 'red']
    labels = [f't = {t_start:.1f}',
              f't = {(t_start + t_end)/2:.1f}',
              f't = {t_end - (t_end - t_start)/4:.1f}',
              f't = {t_end:.1f}']
    
    for i, time_idx in enumerate(time_indices):
        contour_x, contour_y = body.get_deformed_contour(time_idx)
        if len(contour_x) > 0:
            ax2.plot(contour_x, contour_y, color=colors[i], 
                    linewidth=2, alpha=0.7, label=labels[i])
            ax2.fill(contour_x, contour_y, color=colors[i], alpha=0.1)
    
    ax2.set_title(f"Эволюция формы {body_name}", fontsize=14)
    ax2.set_xlabel("Координата x₁", fontsize=12)
    ax2.set_ylabel("Координата x₂", fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    # ========== ШАГ 6: ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ ==========
    print("\n" + "=" * 60)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ:")
    print("=" * 60)
    
    # Вычисляем характеристики деформации
    total_displacement = 0.0
    max_displacement = 0.0
    
    for point in body.points:
        # Смещение от начальной до конечной позиции
        dx = point.trajectory_x[-1] - point.trajectory_x[0]
        dy = point.trajectory_y[-1] - point.trajectory_y[0]
        displacement = math.sqrt(dx**2 + dy**2)
        
        total_displacement += displacement
        max_displacement = max(max_displacement, displacement)
    
    avg_displacement = total_displacement / len(body.points)
    
    print(f"\nСТАТИСТИКА ДВИЖЕНИЯ:")
    print(f"  Среднее смещение точки: {avg_displacement:.4f}")
    print(f"  Максимальное смещение: {max_displacement:.4f}")
    
    # Анализ изменения размеров
    if body.type == BodyType.CIRCLE:
        # Для круга: находим "радиусы" по осям
        initial_center_x = -1.5  # center_x = -size - 0.5 = -1 - 0.5 = -1.5
        initial_center_y = 1.5   # center_y = size + 0.5 = 1 + 0.5 = 1.5
        
        # Находим точки с максимальным/минимальным x и y в конечном состоянии
        final_x = [p.trajectory_x[-1] for p in body.points]
        final_y = [p.trajectory_y[-1] for p in body.points]
        
        # Приближенные "радиусы" деформированного эллипса
        horizontal_extent = max(final_x) - min(final_x)
        vertical_extent = max(final_y) - min(final_y)
        
        print(f"\nДЕФОРМАЦИЯ КРУГА:")
        print(f"  Начальный диаметр: {2.0:.2f}")
        print(f"  Горизонтальный размер эллипса: {horizontal_extent:.3f}")
        print(f"  Вертикальный размер эллипса: {vertical_extent:.3f}")
        print(f"  Отношение осей (гориз/верт): {horizontal_extent/vertical_extent:.3f}")
        
    else:  # SQUARE
        # Для квадрата: находим длины сторон
        # Берем точки на противоположных сторонах
        points_per_side = body.points_per_side
        
        # Нижняя и верхняя стороны
        bottom_points = body.points[:points_per_side]
        top_points = body.points[2*points_per_side:3*points_per_side]
        
        # Левая и правая стороны
        left_points = body.points[3*points_per_side:]
        right_points = body.points[points_per_side:2*points_per_side]
        
        # Вычисляем длины сторон в конечном состоянии
        bottom_length = 0
        top_length = 0
        left_length = 0
        right_length = 0
        
        if len(bottom_points) > 1:
            bottom_length = abs(bottom_points[-1].trajectory_x[-1] - bottom_points[0].trajectory_x[-1])
        
        if len(top_points) > 1:
            top_length = abs(top_points[-1].trajectory_x[-1] - top_points[0].trajectory_x[-1])
        
        if len(left_points) > 1:
            left_length = abs(left_points[-1].trajectory_y[-1] - left_points[0].trajectory_y[-1])
        
        if len(right_points) > 1:
            right_length = abs(right_points[-1].trajectory_y[-1] - right_points[0].trajectory_y[-1])
        
        avg_horizontal = (bottom_length + top_length) / 2
        avg_vertical = (left_length + right_length) / 2
        
        print(f"\nДЕФОРМАЦИЯ КВАДРАТА:")
        print(f"  Начальная сторона: {1.0:.2f}")
        print(f"  Средняя горизонтальная сторона: {avg_horizontal:.3f}")
        print(f"  Средняя вертикальная сторона: {avg_vertical:.3f}")
        print(f"  Отношение сторон (гориз/верт): {avg_horizontal/avg_vertical:.3f}")
    
    # Физическая интерпретация
    print(f"\nФИЗИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ:")
    print(f"Поле скоростей: v₁ = -ln(t)·x₁, v₂ = -e^t·x₂")
    print(f"1. При t > 1: ln(t) > 0, поэтому v₁ направлена ПРОТИВ x₁")
    print(f"   Точки с x₁ < 0 (2-я четверть) движутся ВПРАВО (v₁ > 0)")
    print(f"2. Всегда: e^t > 0, поэтому -e^t < 0, v₂ направлена ПРОТИВ x₂")
    print(f"   Точки с x₂ > 0 (2-я четверть) движутся ВНИЗ (v₂ < 0)")
    
    print("\n" + "=" * 60)
    print("ПРОГРАММА ЗАВЕРШЕНА!")
    print("=" * 60)
