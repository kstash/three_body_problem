import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
from typing import List

def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.sqrt(np.sum((p1 - p2)**2))

def three_body_equations(t: float, y: np.ndarray, masses: List[float]) -> np.ndarray:
    n = len(masses)
    positions = y[:3*n].reshape((n, 3))
    velocities = y[3*n:].reshape((n, 3))
    derivatives = np.zeros_like(y)
    
    derivatives[:3*n] = velocities.reshape(3*n)
    
    for i in range(n):
        acceleration = np.zeros(3)
        for j in range(n):
            if i != j:
                r = distance(positions[i], positions[j])
                acceleration += masses[j] * (positions[j] - positions[i]) / r**3
        derivatives[3*n + 3*i:3*n + 3*(i+1)] = acceleration
    return derivatives

masses = [64, 8, 1]  # 질량
colors = ['blue', 'green', 'red']  # 색상
initial_positions = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0]])
initial_velocities = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 1]])
initial_conditions = np.concatenate([initial_positions.flatten(), initial_velocities.flatten()])

t_span = (0, 5000)
t_eval = np.linspace(*t_span, 50000)
solution = solve_ivp(three_body_equations, t_span, initial_conditions, args=(masses,), t_eval=t_eval, rtol=1e-5)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

lines = [ax.plot([], [], [], lw=2, color=colors[i])[0] for i in range(len(masses))]
points = [ax.plot([], [], [], 'o', markersize=(masses[i]**(1/3)), color=colors[i])[0] for i in range(len(masses))]

def init() -> List:
    for line, point in zip(lines, points):
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
    return lines + points

def update(frame: int) -> List:
    min_val, max_val = np.inf, -np.inf
    for i, (line, point) in enumerate(zip(lines, points)):
        x_data, y_data, z_data = solution.y[i*3, :frame+1], solution.y[i*3+1, :frame+1], solution.y[i*3+2, :frame+1]
        line.set_data(x_data, y_data)
        line.set_3d_properties(z_data)
        point.set_data([x_data[-1]], [y_data[-1]])
        point.set_3d_properties([z_data[-1]])
        # 축 범위 조정
        min_val = min(min_val, np.min(x_data), np.min(y_data), np.min(z_data))
        max_val = max(max_val, np.max(x_data), np.max(y_data), np.max(z_data))
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_zlim(min_val, max_val)
    return lines + points

ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=False, interval=10)

# 정지 및 재시작 버튼
def pause_animation(event):
    ani.event_source.stop()

def start_animation(event):
    ani.event_source.start()

pause_ax = fig.add_axes([0.7, 0.05, 0.1, 0.075])
pause_button = Button(pause_ax, 'Pause')
pause_button.on_clicked(pause_animation)

start_ax = fig.add_axes([0.81, 0.05, 0.1, 0.075])
start_button = Button(start_ax, 'Start')
start_button.on_clicked(start_animation)

plt.show()
