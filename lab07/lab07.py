"""Решение эллиптического уравнения методами итераций.

Задача:
    u_xx + u_yy + u = 0,   (x,y) ∈ [0, π/2] × [0, 1]
    
    Граничные условия:
    u(0, y) = 0           (Дирихле)
    u(π/2, y) = y         (Дирихле)
    u_y(x, 0) = sin(x)    (Неймана)
    u_y(x, 1) - u(x, 1) = 0   (Робина)

Аналитическое решение: u(x, y) = y * sin(x)

Программа реализует три итерационных метода:
- Метод Либмана (простых итераций)
- Метод Зейделя
- Метод верхней релаксации (SOR)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class Method(Enum):
    """Тип итерационного метода."""
    LIEBMANN = "liebmann"
    SEIDEL = "seidel"
    SOR = "sor"


ScalarFunc = Callable[[np.ndarray | float, np.ndarray | float], np.ndarray | float]


@dataclass
class BoundaryCondition:
    """Смешанное граничное условие: α·∂u/∂n + β·u = φ(s).
    
    Частные случаи:
    - α=0, β≠0: условие Дирихле (u = φ/β)
    - α≠0, β=0: условие Неймана (∂u/∂n = φ/α)
    - α≠0, β≠0: условие Робина
    """
    alpha: float  # коэффициент при производной
    beta: float   # коэффициент при функции
    phi: Callable[[np.ndarray | float], np.ndarray | float]  # правая часть


@dataclass
class ProblemParams:
    """Параметры задачи."""
    # Коэффициенты уравнения a·u_xx + b·u_yy + c·u = f(x,y)
    a: float
    b: float
    c: float
    f: ScalarFunc  # правая часть
    
    # Граничные условия
    bc_left: BoundaryCondition    # x = 0
    bc_right: BoundaryCondition   # x = Lx
    bc_bottom: BoundaryCondition  # y = 0
    bc_top: BoundaryCondition     # y = Ly
    
    # Размеры области
    Lx: float
    Ly: float
    
    # Аналитическое решение
    u_exact: ScalarFunc


def build_grid(Lx: float, Ly: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Создаёт равномерную сетку по x и y.
    
    Args:
        Lx: длина области по x
        Ly: длина области по y
        nx: число интервалов по x
        ny: число интервалов по y
    
    Returns:
        x: координаты узлов по x
        y: координаты узлов по y
        hx: шаг сетки по x
        hy: шаг сетки по y
    """
    hx = Lx / nx
    hy = Ly / ny
    x = np.linspace(0.0, Lx, nx + 1)
    y = np.linspace(0.0, Ly, ny + 1)
    return x, y, hx, hy


def compute_scheme_coefficients(hx: float, hy: float, params: ProblemParams) -> Dict[str, float]:
    """Вычисляет коэффициенты центрально-разностной схемы.
    
    Разностная аппроксимация:
    a·(u[i+1,j] - 2u[i,j] + u[i-1,j])/hx² + 
    b·(u[i,j+1] - 2u[i,j] + u[i,j-1])/hy² + 
    c·u[i,j] = f[i,j]
    
    Returns:
        Словарь с коэффициентами: gamma, alpha_minus, alpha_plus, beta_minus, beta_plus
    """
    hx2 = hx * hx
    hy2 = hy * hy
    
    # Знаменатель для нормализации
    gamma = 2.0 * params.a / hx2 + 2.0 * params.b / hy2 - params.c
    
    # Коэффициенты для внутренних узлов
    alpha_minus = params.a / (hx2 * gamma)  # u[i+1,j]
    alpha_plus = params.a / (hx2 * gamma)   # u[i-1,j]
    beta_minus = params.b / (hy2 * gamma)   # u[i,j+1]
    beta_plus = params.b / (hy2 * gamma)    # u[i,j-1]
    
    return {
        "gamma": gamma,
        "alpha_minus": alpha_minus,
        "alpha_plus": alpha_plus,
        "beta_minus": beta_minus,
        "beta_plus": beta_plus,
    }


def apply_boundary_conditions(
    u: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    hx: float,
    hy: float,
    params: ProblemParams,
) -> None:
    """Применяет граничные условия всех типов (Дирихле, Неймана, Робина).
    
    Модифицирует массив u на границах области.
    """
    nx = len(x) - 1
    ny = len(y) - 1
    
    # Левая граница (x = 0)
    bc = params.bc_left
    if bc.alpha == 0:  # Дирихле
        u[:, 0] = bc.phi(y) / bc.beta
    else:  # Неймана или Робина
        # ∂u/∂x ≈ (u[1] - u[0]) / hx
        # α·(u[1] - u[0])/hx + β·u[0] = φ
        # u[0] = (hx·φ - α·u[1]) / (hx·β - α)
        phi_vals = bc.phi(y)
        u[:, 0] = (hx * phi_vals - bc.alpha * u[:, 1]) / (hx * bc.beta - bc.alpha)
    
    # Правая граница (x = Lx)
    bc = params.bc_right
    if bc.alpha == 0:  # Дирихле
        u[:, -1] = bc.phi(y) / bc.beta
    else:  # Неймана или Робина
        # ∂u/∂x ≈ (u[nx] - u[nx-1]) / hx
        # α·(u[nx] - u[nx-1])/hx + β·u[nx] = φ
        # u[nx] = (hx·φ + α·u[nx-1]) / (hx·β + α)
        phi_vals = bc.phi(y)
        u[:, -1] = (hx * phi_vals + bc.alpha * u[:, -2]) / (hx * bc.beta + bc.alpha)
    
    # Нижняя граница (y = 0)
    bc = params.bc_bottom
    if bc.alpha == 0:  # Дирихле
        u[0, :] = bc.phi(x) / bc.beta
    else:  # Неймана или Робина
        # ∂u/∂y ≈ (u[1] - u[0]) / hy
        # α·(u[1] - u[0])/hy + β·u[0] = φ
        # u[0] = (hy·φ - α·u[1]) / (hy·β - α)
        phi_vals = bc.phi(x)
        u[0, :] = (hy * phi_vals - bc.alpha * u[1, :]) / (hy * bc.beta - bc.alpha)
    
    # Верхняя граница (y = Ly)
    bc = params.bc_top
    if bc.alpha == 0:  # Дирихле
        u[-1, :] = bc.phi(x) / bc.beta
    else:  # Неймана или Робина
        # ∂u/∂y ≈ (u[ny] - u[ny-1]) / hy
        # α·(u[ny] - u[ny-1])/hy + β·u[ny] = φ
        # u[ny] = (hy·φ + α·u[ny-1]) / (hy·β + α)
        phi_vals = bc.phi(x)
        u[-1, :] = (hy * phi_vals + bc.alpha * u[-2, :]) / (hy * bc.beta + bc.alpha)


def liebmann_method(
    x: np.ndarray,
    y: np.ndarray,
    hx: float,
    hy: float,
    params: ProblemParams,
    epsilon: float = 1e-4,
    u0: float = 0.0,
    max_iter: int = 100000,
) -> Tuple[np.ndarray, int]:
    """Метод Либмана (простых итераций).
    
    На каждой итерации вычисляются новые значения u_next на основе старых u.
    После обновления всех узлов проверяется сходимость.
    
    Returns:
        u: численное решение
        iterations: количество итераций
    """
    nx = len(x) - 1
    ny = len(y) - 1
    
    # Инициализация
    u = np.full((ny + 1, nx + 1), u0, dtype=float)
    u_next = np.copy(u)
    
    # Коэффициенты схемы
    coeff = compute_scheme_coefficients(hx, hy, params)
    gamma = coeff["gamma"]
    alpha_minus = coeff["alpha_minus"]
    alpha_plus = coeff["alpha_plus"]
    beta_minus = coeff["beta_minus"]
    beta_plus = coeff["beta_plus"]
    
    # Итерационный процесс
    for k in range(max_iter):
        max_diff = 0.0
        
        # Обновление внутренних узлов
        for j in range(1, ny):
            for i in range(1, nx):
                u_next[j, i] = (
                    alpha_minus * u[j, i + 1]
                    + alpha_plus * u[j, i - 1]
                    + beta_minus * u[j + 1, i]
                    + beta_plus * u[j - 1, i]
                    - params.f(x[i], y[j]) / gamma
                )
                max_diff = max(max_diff, abs(u_next[j, i] - u[j, i]))
        
        # Обновление граничных узлов
        apply_boundary_conditions(u_next, x, y, hx, hy, params)
        
        # Проверка граничных узлов на изменение
        for i in range(nx + 1):
            max_diff = max(max_diff, abs(u_next[0, i] - u[0, i]))
            max_diff = max(max_diff, abs(u_next[ny, i] - u[ny, i]))
        for j in range(ny + 1):
            max_diff = max(max_diff, abs(u_next[j, 0] - u[j, 0]))
            max_diff = max(max_diff, abs(u_next[j, nx] - u[j, nx]))
        
        # Копирование результата
        u, u_next = u_next, u
        
        # Проверка сходимости
        if max_diff <= epsilon:
            return u, k + 1
    
    print(f"Предупреждение: метод Либмана не сошелся за {max_iter} итераций")
    return u, max_iter


def seidel_method(
    x: np.ndarray,
    y: np.ndarray,
    hx: float,
    hy: float,
    params: ProblemParams,
    epsilon: float = 1e-4,
    u0: float = 0.0,
    max_iter: int = 100000,
) -> Tuple[np.ndarray, int]:
    """Метод Зейделя.
    
    Отличается от метода Либмана тем, что новые значения используются сразу
    после их вычисления (in-place обновление).
    
    Returns:
        u: численное решение
        iterations: количество итераций
    """
    nx = len(x) - 1
    ny = len(y) - 1
    
    # Инициализация
    u = np.full((ny + 1, nx + 1), u0, dtype=float)
    u_prev = np.copy(u)
    
    # Коэффициенты схемы
    coeff = compute_scheme_coefficients(hx, hy, params)
    gamma = coeff["gamma"]
    alpha_minus = coeff["alpha_minus"]
    alpha_plus = coeff["alpha_plus"]
    beta_minus = coeff["beta_minus"]
    beta_plus = coeff["beta_plus"]
    
    # Итерационный процесс
    for k in range(max_iter):
        max_diff = 0.0
        
        # Обновление внутренних узлов (используем новые значения сразу)
        for j in range(1, ny):
            for i in range(1, nx):
                u_new = (
                    alpha_minus * u[j, i + 1]
                    + alpha_plus * u[j, i - 1]
                    + beta_minus * u[j + 1, i]
                    + beta_plus * u[j - 1, i]
                    - params.f(x[i], y[j]) / gamma
                )
                max_diff = max(max_diff, abs(u_new - u[j, i]))
                u[j, i] = u_new
        
        # Обновление граничных узлов
        u_prev[:, :] = u[:, :]
        apply_boundary_conditions(u, x, y, hx, hy, params)
        
        # Проверка граничных узлов на изменение
        for i in range(nx + 1):
            max_diff = max(max_diff, abs(u[0, i] - u_prev[0, i]))
            max_diff = max(max_diff, abs(u[ny, i] - u_prev[ny, i]))
        for j in range(ny + 1):
            max_diff = max(max_diff, abs(u[j, 0] - u_prev[j, 0]))
            max_diff = max(max_diff, abs(u[j, nx] - u_prev[j, nx]))
        
        # Проверка сходимости
        if max_diff <= epsilon:
            return u, k + 1
    
    print(f"Предупреждение: метод Зейделя не сошелся за {max_iter} итераций")
    return u, max_iter


def sor_method(
    x: np.ndarray,
    y: np.ndarray,
    hx: float,
    hy: float,
    params: ProblemParams,
    epsilon: float = 1e-4,
    u0: float = 0.0,
    max_iter: int = 100000,
    omega: float = 1.5,
) -> Tuple[np.ndarray, int]:
    """Метод последовательной верхней релаксации (SOR).
    
    Использует релаксацию: u_new = u_old + ω·(u_iter - u_old)
    где ω - параметр релаксации (обычно 1 < ω < 2).
    
    Args:
        omega: параметр релаксации
    
    Returns:
        u: численное решение
        iterations: количество итераций
    """
    nx = len(x) - 1
    ny = len(y) - 1
    
    # Инициализация
    u = np.full((ny + 1, nx + 1), u0, dtype=float)
    u_prev = np.copy(u)
    
    # Коэффициенты схемы
    coeff = compute_scheme_coefficients(hx, hy, params)
    gamma = coeff["gamma"]
    alpha_minus = coeff["alpha_minus"]
    alpha_plus = coeff["alpha_plus"]
    beta_minus = coeff["beta_minus"]
    beta_plus = coeff["beta_plus"]
    
    # Итерационный процесс
    for k in range(max_iter):
        max_diff = 0.0
        
        # Обновление внутренних узлов с релаксацией
        for j in range(1, ny):
            for i in range(1, nx):
                u_iter = (
                    alpha_minus * u[j, i + 1]
                    + alpha_plus * u[j, i - 1]
                    + beta_minus * u[j + 1, i]
                    + beta_plus * u[j - 1, i]
                    - params.f(x[i], y[j]) / gamma
                )
                u_new = u[j, i] + omega * (u_iter - u[j, i])
                max_diff = max(max_diff, abs(u_new - u[j, i]))
                u[j, i] = u_new
        
        # Обновление граничных узлов с релаксацией
        u_prev[:, :] = u[:, :]
        u_temp = np.copy(u)
        apply_boundary_conditions(u_temp, x, y, hx, hy, params)
        
        # Применение релаксации к граничным узлам
        u[:, :] = u_prev + omega * (u_temp - u_prev)
        
        # Проверка граничных узлов на изменение
        for i in range(nx + 1):
            max_diff = max(max_diff, abs(u[0, i] - u_prev[0, i]))
            max_diff = max(max_diff, abs(u[ny, i] - u_prev[ny, i]))
        for j in range(ny + 1):
            max_diff = max(max_diff, abs(u[j, 0] - u_prev[j, 0]))
            max_diff = max(max_diff, abs(u[j, nx] - u_prev[j, nx]))
        
        # Проверка сходимости
        if max_diff <= epsilon:
            return u, k + 1
    
    print(f"Предупреждение: метод SOR не сошелся за {max_iter} итераций")
    return u, max_iter


def compute_errors(
    u: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    hx: float,
    hy: float,
    u_exact: ScalarFunc,
) -> Dict[str, float]:
    """Вычисляет различные нормы погрешности.
    
    Returns:
        Словарь с max_error, l2_error, mean_error
    """
    X, Y = np.meshgrid(x, y)
    u_exact_vals = u_exact(X, Y)
    diff = np.abs(u - u_exact_vals)
    
    return {
        "max_error": float(np.max(diff)),
        "l2_error": float(np.sqrt(hx * hy * np.sum(diff**2))),
        "mean_error": float(np.mean(diff)),
    }


def print_error_table(method_name: str, errors: Dict[str, float], iterations: int) -> None:
    """Выводит таблицу с погрешностями."""
    print(f"\n{method_name}:")
    print(f"  Итераций:        {iterations}")
    print(f"  Max погрешность: {errors['max_error']:.6e}")
    print(f"  L2 погрешность:  {errors['l2_error']:.6e}")
    print(f"  Ср. погрешность: {errors['mean_error']:.6e}")


def plot_2d_heatmaps(
    x: np.ndarray,
    y: np.ndarray,
    u_numerical: np.ndarray,
    u_exact: ScalarFunc,
    method: Method,
    iterations: int,
) -> None:
    """Создаёт три тепловые карты: численное решение, точное решение, погрешность."""
    X, Y = np.meshgrid(x, y)
    u_exact_vals = u_exact(X, Y)
    error = np.abs(u_numerical - u_exact_vals)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    # Численное решение
    im1 = axes[0].imshow(u_numerical, extent=[0, x[-1], 0, y[-1]], origin='lower', 
                         aspect='auto', cmap='viridis')
    axes[0].set_title(f'Численное решение\n{method.value}, итераций: {iterations}')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    # Аналитическое решение
    im2 = axes[1].imshow(u_exact_vals, extent=[0, x[-1], 0, y[-1]], origin='lower',
                         aspect='auto', cmap='viridis')
    axes[1].set_title('Аналитическое решение')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1])
    
    # Абсолютная погрешность
    im3 = axes[2].imshow(error, extent=[0, x[-1], 0, y[-1]], origin='lower',
                         aspect='auto', cmap='OrRd')
    axes[2].set_title(f'Абсолютная погрешность\nmax = {np.max(error):.3e}')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(f'heatmap_{method.value}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_3d_surfaces(
    x: np.ndarray,
    y: np.ndarray,
    u_numerical: np.ndarray,
    u_exact: ScalarFunc,
    method: Method,
) -> None:
    """Создаёт 3D графики для численного и аналитического решений."""
    X, Y = np.meshgrid(x, y)
    U_exact = u_exact(X, Y)
    
    fig = plt.figure(figsize=(16, 6))
    
    # 3D график аналитического решения
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, Y, U_exact, cmap='viridis', alpha=0.9, edgecolor='none')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u(x,y)')
    ax1.set_title('Аналитическое решение')
    ax1.view_init(elev=25, azim=45)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    
    # 3D график численного решения
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, Y, u_numerical, cmap='plasma', alpha=0.9, edgecolor='none')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u(x,y)')
    ax2.set_title(f'Численное решение ({method.value})')
    ax2.view_init(elev=25, azim=45)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.savefig(f'3d_surface_{method.value}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_cross_sections(
    x: np.ndarray,
    y: np.ndarray,
    u_numerical: np.ndarray,
    u_exact: ScalarFunc,
    method: Method,
) -> None:
    """Строит срезы решения по x и y для визуального сравнения."""
    X, Y = np.meshgrid(x, y)
    U_exact = u_exact(X, Y)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Срез по x (при y = Ly/2)
    j_mid = len(y) // 2
    axes[0].plot(x, u_numerical[j_mid, :], 'b-', label='численное', linewidth=2)
    axes[0].plot(x, U_exact[j_mid, :], 'r--', label='аналитическое', linewidth=1.5)
    axes[0].set_title(f'Срез при y = {y[j_mid]:.3f}')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Срез по y (при x = Lx/2)
    i_mid = len(x) // 2
    axes[1].plot(y, u_numerical[:, i_mid], 'b-', label='численное', linewidth=2)
    axes[1].plot(y, U_exact[:, i_mid], 'r--', label='аналитическое', linewidth=1.5)
    axes[1].set_title(f'Срез при x = {x[i_mid]:.3f}')
    axes[1].set_xlabel('y')
    axes[1].set_ylabel('u')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'cross_sections_{method.value}.png', dpi=150, bbox_inches='tight')
    plt.close()


def explore_grid_sensitivity(
    params: ProblemParams,
    method: Method,
    omega: float | None = None,
) -> None:
    """Исследует зависимость погрешности от параметров сетки."""
    # Варианты размеров сетки
    grid_sizes = [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)]
    epsilon = 1e-4
    
    max_errors = []
    l2_errors = []
    iterations_list = []
    h_values = []
    
    print(f"\n{'='*70}")
    print(f"Зависимость погрешности от размера сетки ({method.value})")
    if omega is not None:
        print(f"Параметр релаксации ω = {omega}")
    print(f"{'nx':>6} {'ny':>6} {'h':>10} {'max_error':>12} {'l2_error':>12} {'iter':>8}")
    print("-" * 70)
    
    for nx, ny in grid_sizes:
        x, y, hx, hy = build_grid(params.Lx, params.Ly, nx, ny)
        h_avg = (hx + hy) / 2
        h_values.append(h_avg)
        
        if method == Method.LIEBMANN:
            u, iters = liebmann_method(x, y, hx, hy, params, epsilon)
        elif method == Method.SEIDEL:
            u, iters = seidel_method(x, y, hx, hy, params, epsilon)
        else:  # SOR
            u, iters = sor_method(x, y, hx, hy, params, epsilon, omega=omega)
        
        errors = compute_errors(u, x, y, hx, hy, params.u_exact)
        max_errors.append(errors['max_error'])
        l2_errors.append(errors['l2_error'])
        iterations_list.append(iters)
        
        print(f"{nx:6d} {ny:6d} {h_avg:10.6f} {errors['max_error']:12.4e} "
              f"{errors['l2_error']:12.4e} {iters:8d}")
    
    # Графики зависимости ошибок от h
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Max error
    axes[0].loglog(h_values, max_errors, 'o-', label='max error', linewidth=2)
    axes[0].set_xlabel('h (средний шаг сетки)')
    axes[0].set_ylabel('max error')
    axes[0].set_title(f'Максимальная погрешность\n{method.value}')
    axes[0].grid(True, which='both', alpha=0.3)
    axes[0].legend()
    
    # L2 error
    axes[1].loglog(h_values, l2_errors, 's-', label='L2 error', color='orange', linewidth=2)
    axes[1].set_xlabel('h (средний шаг сетки)')
    axes[1].set_ylabel('L2 error')
    axes[1].set_title(f'L2 погрешность\n{method.value}')
    axes[1].grid(True, which='both', alpha=0.3)
    axes[1].legend()
    
    # Iterations
    axes[2].plot(h_values, iterations_list, '^-', label='iterations', color='green', linewidth=2)
    axes[2].set_xlabel('h (средний шаг сетки)')
    axes[2].set_ylabel('Число итераций')
    axes[2].set_title(f'Сходимость\n{method.value}')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].invert_xaxis()  # Меньший h -> больше итераций
    
    plt.tight_layout()
    method_label = f"{method.value}_omega{omega:.2f}" if omega else method.value
    plt.savefig(f'grid_sensitivity_{method_label}.png', dpi=150, bbox_inches='tight')
    plt.close()


def compare_methods_iterations(results: Dict[str, Tuple[np.ndarray, int, Dict[str, float]]]) -> None:
    """Сравнивает результаты разных методов."""
    methods = list(results.keys())
    iterations = [results[m][1] for m in methods]
    max_errors = [results[m][2]['max_error'] for m in methods]
    l2_errors = [results[m][2]['l2_error'] for m in methods]
    
    # Таблица с результатами
    print(f"\n{'='*70}")
    print("Сравнение методов")
    print(f"{'Метод':20} {'Итераций':>10} {'Max error':>15} {'L2 error':>15}")
    print("-" * 70)
    for method in methods:
        u, iters, errs = results[method]
        print(f"{method:20} {iters:10d} {errs['max_error']:15.6e} {errs['l2_error']:15.6e}")
    
    # Столбчатая диаграмма итераций
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Итерации
    axes[0].bar(methods, iterations, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0].set_ylabel('Число итераций')
    axes[0].set_title('Сравнение скорости сходимости')
    axes[0].tick_params(axis='x', rotation=15)
    for i, v in enumerate(iterations):
        axes[0].text(i, v + max(iterations) * 0.02, str(v), ha='center', va='bottom')
    
    # Max errors
    axes[1].bar(methods, max_errors, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1].set_ylabel('Max error')
    axes[1].set_title('Максимальная погрешность')
    axes[1].set_yscale('log')
    axes[1].tick_params(axis='x', rotation=15)
    
    # L2 errors
    axes[2].bar(methods, l2_errors, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[2].set_ylabel('L2 error')
    axes[2].set_title('L2 погрешность')
    axes[2].set_yscale('log')
    axes[2].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig('method_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def run_experiment() -> None:
    """Основной эксперимент: решение задачи всеми методами с анализом."""
    
    # ========================================================================
    # ПАРАМЕТРЫ ЗАДАЧИ
    # ========================================================================
    
    print("="*70)
    print("Решение эллиптического уравнения итерационными методами")
    print("="*70)
    print("\nУравнение: u_xx + u_yy + u = 0")
    print("Область: [0, π/2] × [0, 1]")
    print("\nГраничные условия:")
    print("  u(0, y) = 0           (Дирихле)")
    print("  u(π/2, y) = y         (Дирихле)")
    print("  u_y(x, 0) = sin(x)    (Неймана)")
    print("  u_y(x, 1) - u(x, 1) = 0   (Робина)")
    print("\nАналитическое решение: u(x,y) = y·sin(x)")
    
    # Коэффициенты уравнения
    a = 1.0
    b = 1.0
    c = 1.0
    
    def f(x: np.ndarray | float, y: np.ndarray | float):
        """Правая часть уравнения."""
        return 0.0 * x * y  # Возвращает нули нужной формы
    
    # Аналитическое решение
    def u_exact(x: np.ndarray | float, y: np.ndarray | float):
        """Точное решение u(x, y) = y·sin(x)."""
        return y * np.sin(x)
    
    # Граничные условия
    bc_left = BoundaryCondition(
        alpha=0.0,
        beta=1.0,
        phi=lambda y: 0.0 * y,  # u(0, y) = 0
    )
    
    bc_right = BoundaryCondition(
        alpha=0.0,
        beta=1.0,
        phi=lambda y: y,  # u(π/2, y) = y
    )
    
    bc_bottom = BoundaryCondition(
        alpha=1.0,
        beta=0.0,
        phi=lambda x: np.sin(x),  # u_y(x, 0) = sin(x)
    )
    
    bc_top = BoundaryCondition(
        alpha=1.0,
        beta=-1.0,
        phi=lambda x: 0.0 * x,  # u_y(x, 1) - u(x, 1) = 0
    )
    
    # Параметры области
    Lx = np.pi / 2
    Ly = 1.0
    
    # Параметры сетки
    nx = 35
    ny = 35
    
    # Параметры итераций
    epsilon = 1e-6
    u0 = 2.0
    omega = 1.75
    
    # ========================================================================
    # КОНЕЦ ПАРАМЕТРОВ
    # ========================================================================
    
    params = ProblemParams(
        a=a, b=b, c=c, f=f,
        bc_left=bc_left,
        bc_right=bc_right,
        bc_bottom=bc_bottom,
        bc_top=bc_top,
        Lx=Lx,
        Ly=Ly,
        u_exact=u_exact,
    )
    
    x, y, hx, hy = build_grid(Lx, Ly, nx, ny)
    
    print(f"\nПараметры сетки:")
    print(f"  nx = {nx}, ny = {ny}")
    print(f"  hx = {hx:.6f}, hy = {hy:.6f}")
    print(f"  Точность: ε = {epsilon}")
    print(f"  Начальное приближение: u₀ = {u0}")
    print(f"  Параметр релаксации (SOR): ω = {omega}")
    
    # ========================================================================
    # РЕШЕНИЕ ВСЕМИ МЕТОДАМИ
    # ========================================================================
    
    results = {}
    
    # Метод Либмана
    print(f"\n{'='*70}")
    print("Метод Либмана (простых итераций)")
    print("-" * 70)
    u_liebmann, iter_liebmann = liebmann_method(x, y, hx, hy, params, epsilon, u0)
    err_liebmann = compute_errors(u_liebmann, x, y, hx, hy, params.u_exact)
    print_error_table("Метод Либмана", err_liebmann, iter_liebmann)
    results["Либман"] = (u_liebmann, iter_liebmann, err_liebmann)
    
    # Метод Зейделя
    print(f"\n{'='*70}")
    print("Метод Зейделя")
    print("-" * 70)
    u_seidel, iter_seidel = seidel_method(x, y, hx, hy, params, epsilon, u0)
    err_seidel = compute_errors(u_seidel, x, y, hx, hy, params.u_exact)
    print_error_table("Метод Зейделя", err_seidel, iter_seidel)
    results["Зейдель"] = (u_seidel, iter_seidel, err_seidel)
    
    # Метод SOR
    print(f"\n{'='*70}")
    print(f"Метод верхней релаксации (SOR) с ω = {omega}")
    print("-" * 70)
    u_sor, iter_sor = sor_method(x, y, hx, hy, params, epsilon, u0, omega=omega)
    err_sor = compute_errors(u_sor, x, y, hx, hy, params.u_exact)
    print_error_table(f"Метод SOR (ω={omega})", err_sor, iter_sor)
    results[f"SOR (ω={omega})"] = (u_sor, iter_sor, err_sor)
    
    # ========================================================================
    # ВИЗУАЛИЗАЦИЯ
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("Создание графиков...")
    print("-" * 70)
    
    # 2D тепловые карты для каждого метода
    print("  - Тепловые карты...")
    plot_2d_heatmaps(x, y, u_liebmann, params.u_exact, Method.LIEBMANN, iter_liebmann)
    plot_2d_heatmaps(x, y, u_seidel, params.u_exact, Method.SEIDEL, iter_seidel)
    plot_2d_heatmaps(x, y, u_sor, params.u_exact, Method.SOR, iter_sor)
    
    # 3D поверхности для каждого метода
    print("  - 3D поверхности...")
    plot_3d_surfaces(x, y, u_liebmann, params.u_exact, Method.LIEBMANN)
    plot_3d_surfaces(x, y, u_seidel, params.u_exact, Method.SEIDEL)
    plot_3d_surfaces(x, y, u_sor, params.u_exact, Method.SOR)
    
    # Срезы для каждого метода
    print("  - Срезы решений...")
    plot_cross_sections(x, y, u_liebmann, params.u_exact, Method.LIEBMANN)
    plot_cross_sections(x, y, u_seidel, params.u_exact, Method.SEIDEL)
    plot_cross_sections(x, y, u_sor, params.u_exact, Method.SOR)
    
    # Сравнение методов
    print("  - Сравнение методов...")
    compare_methods_iterations(results)
    
    # ========================================================================
    # АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ К СЕТКЕ
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("Анализ зависимости погрешности от размера сетки")
    print("="*70)
    
    explore_grid_sensitivity(params, Method.LIEBMANN)
    explore_grid_sensitivity(params, Method.SEIDEL)
    explore_grid_sensitivity(params, Method.SOR, omega=omega)
    
    print(f"\n{'='*70}")
    print("Эксперимент завершён!")
    print("Все графики сохранены в текущей директории.")
    print("="*70)


if __name__ == "__main__":
    run_experiment()

