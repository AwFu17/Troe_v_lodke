import numpy as np
import matplotlib.pyplot as plt

class Visualisation():
    def __init__(self, n, x1_min, x1_max, x2_min, x2_max):
        self.n = n
        self.x1_min = x1_min
        self.x1_max = x1_max
        self.x2_min = x2_min
        self.x2_max = x2_max

    def visualisate(self, t):
        x1 = np.linspace(self.x1_min, self.x1_max, self.n)
        x2 = np.linspace(self.x2_min, self.x2_max, self.n)
        X1, X2 = np.meshgrid(x1, x2)
        V1 = -X1 * np.log(t)
        V2 = -X2 * np.exp(t)



        # СОЗДАЕМ ФИГУРУ С ДВУМЯ ПОДГРАФИКАМИ ГОРИЗОНТАЛЬНО
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # --- ЛЕВЫЙ ГРАФИК: Векторное поле ---
        mag = np.sqrt(V1 ** 2 + V2 ** 2)
        q = ax1.quiver(X1, X2, V1 / mag, V2 / mag, mag, scale=20, cmap='plasma')
        plt.colorbar(q, ax=ax1)  # colorbar для левого графика

        # Настройка левого графика
        ax1.set_title('Векторное поле скоростей', fontsize=14)
        ax1.set_xlabel('X1', fontsize=12)
        ax1.set_ylabel('X2', fontsize=12)
        ax1.axis('equal')
        ax1.grid(True, alpha=0.3)

        # --- ПРАВЫЙ ГРАФИК: Линии тока ---
        ax2.streamplot(X1, X2, V1, V2,
                                linewidth=1.4,
                                density=self.n/15,
                                arrowsize=1.0,
                                color='black')

        # Настройка правого графика
        ax2.set_title('Линии тока', fontsize=14)
        ax2.set_xlabel('X1', fontsize=12)
        ax2.set_ylabel('X2', fontsize=12)
        ax2.axis('equal')
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f'Визуализация при t={t:.2f}', fontsize=10, y=0.05)


        plt.tight_layout()
        plt.show()


