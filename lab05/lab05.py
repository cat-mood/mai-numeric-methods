"""Решение параболического уравнения теплопроводности разными разностными схемами.

Задача:
    u_t = u_xx + cos(x) * (cos(t) + sin(t)),   0 < x < L,   0 < t <= T
    u(0, t) = sin(t)
    u_x(L, t) = -sin(t)
    u(x, 0) = 0

Аналитическое решение: u(x, t) = sin(t) * cos(x)

Программа демонстрирует три схемы (явная, неявная, Кранка–Николсона)
и три аппроксимации граничного условия с производной.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class Scheme(Enum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    CRANK_NICOLSON = "crank_nicolson"


class BCApprox(Enum):
    TWO_POINT_FIRST = "two_point_first"
    THREE_POINT_SECOND = "three_point_second"
    TWO_POINT_SECOND = "two_point_second"


ScalarFunc = Callable[[np.ndarray | float, np.ndarray | float], np.ndarray | float]


@dataclass
class BoundaryCondition:
    alpha: float
    beta: float
    g: Callable[[float], float]


def apply_two_point_second_explicit_boundary(
    u: np.ndarray,
    k: int,
    x: np.ndarray,
    t: np.ndarray,
    h: float,
    tau: float,
    f: ScalarFunc,
    bc_left: BoundaryCondition,
    bc_right: BoundaryCondition,
) -> None:
    """Явная схема с фиктивными узлами для двухточечной аппроксимации второго порядка."""

    m_new = k + 1
    t_prev = t[k]
    t_new = t[m_new]
    eps = 1e-12

    if abs(bc_left.beta) < eps:
        u[m_new, 0] = bc_left.g(t_new) / bc_left.alpha
    else:
        g_prev = bc_left.g(t_prev)
        alpha = bc_left.alpha
        beta = bc_left.beta

        # Фиктивная точка на старом слое
        u_minus_prev = u[k, 1] - 2.0 * h * (g_prev - alpha * u[k, 0]) / beta

        # Явное обновление узла x=0 с использованием фиктивной точки
        u[m_new, 0] = u[k, 0] + tau * (
            (u_minus_prev - 2.0 * u[k, 0] + u[k, 1]) / h**2
            + f(x[0], t_prev)
        )

    if abs(bc_right.beta) < eps:
        u[m_new, -1] = bc_right.g(t_new) / bc_right.alpha
    else:
        g_prev = bc_right.g(t_prev)
        alpha = bc_right.alpha
        beta = bc_right.beta

        # Фиктивная точка на старом слое
        u_plus_prev = u[k, -2] + 2.0 * h * (g_prev - alpha * u[k, -1]) / beta

        # Явное обновление узла x=L с использованием фиктивной точки
        u[m_new, -1] = u[k, -1] + tau * (
            (u[k, -2] - 2.0 * u[k, -1] + u_plus_prev) / h**2
            + f(x[-1], t_prev)
        )


def build_grid(L: float, T: float, h: float, tau: float) -> Tuple[np.ndarray, np.ndarray]:
    """Создаёт равномерную сетку по x и t."""

    N = int(round(L / h))
    M = int(round(T / tau))
    x = np.linspace(0.0, L, N + 1)
    t = np.linspace(0.0, T, M + 1)
    return x, t


def explicit_scheme(
    x: np.ndarray,
    t: np.ndarray,
    h: float,
    tau: float,
    f: ScalarFunc,
    u0: Callable[[np.ndarray], np.ndarray],
    bc_left: BoundaryCondition,
    bc_right: BoundaryCondition,
    bc_approx: BCApprox,
) -> np.ndarray:
    """Явная схема."""

    sigma = tau / h**2
    if sigma > 0.5:
        print(
            f"[предупреждение] Для явной схемы лучше взять меньший шаг: sigma = {sigma:.3f} > 0.5"
        )

    M = len(t) - 1
    N = len(x) - 1
    u = np.zeros((M + 1, N + 1))
    u[0, :] = u0(x)

    for k in range(M):
        time_prev = t[k]
        time_new = t[k + 1]

        for j in range(1, N):
            laplace = (u[k, j + 1] - 2.0 * u[k, j] + u[k, j - 1]) / h**2
            u[k + 1, j] = u[k, j] + tau * (laplace + f(x[j], time_prev))

        if bc_approx == BCApprox.TWO_POINT_SECOND:
            apply_two_point_second_explicit_boundary(
                u, k, x, t, h, tau, f, bc_left, bc_right
            )
        else:
            u[k + 1, 0] = resolve_left_boundary(u[k + 1, :], bc_left, bc_approx, h, time_new)
            u[k + 1, -1] = resolve_right_boundary(
                u[k + 1, :], bc_right, bc_approx, h, time_new
            )

    return u


def implicit_scheme(
    x: np.ndarray,
    t: np.ndarray,
    h: float,
    tau: float,
    f: ScalarFunc,
    u0: Callable[[np.ndarray], np.ndarray],
    bc_left: BoundaryCondition,
    bc_right: BoundaryCondition,
    bc_approx: BCApprox,
) -> np.ndarray:
    """Неявная схема."""

    sigma = tau / h**2
    M = len(t) - 1
    N = len(x) - 1
    u = np.zeros((M + 1, N + 1))
    u[0, :] = u0(x)

    for k in range(M):
        time_new = t[k + 1]
        a, b, c, d = build_implicit_system(
            u[k, :], sigma, tau, h, x, time_new, f, bc_left, bc_right, bc_approx
        )
        u[k + 1, :] = solve_tridiagonal(a, b, c, d)

    return u


def crank_nicolson_scheme(
    x: np.ndarray,
    t: np.ndarray,
    h: float,
    tau: float,
    f: ScalarFunc,
    u0: Callable[[np.ndarray], np.ndarray],
    bc_left: BoundaryCondition,
    bc_right: BoundaryCondition,
    bc_approx: BCApprox,
) -> np.ndarray:
    """Схема Кранка–Николсона."""

    sigma = tau / h**2
    M = len(t) - 1
    N = len(x) - 1
    u = np.zeros((M + 1, N + 1))
    u[0, :] = u0(x)

    for k in range(M):
        time_prev = t[k]
        time_new = t[k + 1]
        a, b, c, d = build_cn_system(
            u[k, :], sigma, tau, h, x, time_prev, time_new, f, bc_left, bc_right, bc_approx
        )
        u[k + 1, :] = solve_tridiagonal(a, b, c, d)

    return u


def build_implicit_system(
    u_prev: np.ndarray,
    sigma: float,
    tau: float,
    h: float,
    x: np.ndarray,
    time_new: float,
    f: ScalarFunc,
    bc_left: BoundaryCondition,
    bc_right: BoundaryCondition,
    bc_approx: BCApprox,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Готовит трёхдиагональную СЛАУ для неявной схемы."""

    n = len(u_prev)
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)

    for j in range(1, n - 1):
        a[j] = -sigma
        b[j] = 1.0 + 2.0 * sigma
        c[j] = -sigma
        d[j] = u_prev[j] + tau * f(x[j], time_new)

    eps = 1e-12

    if bc_approx == BCApprox.TWO_POINT_SECOND and abs(bc_left.beta) > eps:
        alpha = bc_left.alpha
        beta = bc_left.beta
        g_new = bc_left.g(time_new)
        rhs = u_prev[0] + tau * f(x[0], time_new)
        b[0] = 1.0 + 2.0 * sigma - (2.0 * sigma * h * alpha / beta)
        c[0] = -2.0 * sigma
        d[0] = rhs - (2.0 * sigma * h / beta) * g_new
    elif bc_approx == BCApprox.THREE_POINT_SECOND and abs(bc_left.beta) > eps:
        alpha = bc_left.alpha
        beta = bc_left.beta
        g_new = bc_left.g(time_new)
        b[0] = -2.0 * sigma
        c[0] = 2.0 * sigma - 1.0
        d[0] = 2.0 * sigma * h * g_new / beta - u_prev[1]
    else:
        apply_left_bc_to_system(
            a, b, c, d, bc_left, bc_approx, sigma, h, time_new, scheme="implicit"
        )

    if bc_approx == BCApprox.TWO_POINT_SECOND and abs(bc_right.beta) > eps:
        alpha = bc_right.alpha
        beta = bc_right.beta
        g_new = bc_right.g(time_new)
        rhs = u_prev[-1] + tau * f(x[-1], time_new)
        a[-1] = -2.0 * sigma
        b[-1] = 1.0 + 2.0 * sigma + (2.0 * sigma * h * alpha / beta)
        d[-1] = rhs + (2.0 * sigma * h / beta) * g_new
    elif bc_approx == BCApprox.THREE_POINT_SECOND and abs(bc_right.beta) > eps:
        alpha = bc_right.alpha
        beta = bc_right.beta
        g_new = bc_right.g(time_new)
        a[-1] = 2.0 * sigma - 1.0
        b[-1] = -2.0 * sigma
        d[-1] = -2.0 * sigma * h * g_new / beta - u_prev[-2]
    else:
        apply_right_bc_to_system(
            a, b, c, d, bc_right, bc_approx, sigma, h, time_new, scheme="implicit"
        )

    return a, b, c, d


def build_cn_system(
    u_prev: np.ndarray,
    sigma: float,
    tau: float,
    h: float,
    x: np.ndarray,
    time_prev: float,
    time_new: float,
    f: ScalarFunc,
    bc_left: BoundaryCondition,
    bc_right: BoundaryCondition,
    bc_approx: BCApprox,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Готовит трёхдиагональную СЛАУ для схемы Кранка–Николсона."""

    n = len(u_prev)
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)

    for j in range(1, n - 1):
        a[j] = -0.5 * sigma
        b[j] = 1.0 + sigma
        c[j] = -0.5 * sigma
        laplace_prev = (u_prev[j + 1] - 2.0 * u_prev[j] + u_prev[j - 1])
        rhs_laplace = 0.5 * sigma * laplace_prev
        rhs_force = 0.5 * tau * (f(x[j], time_prev) + f(x[j], time_new))
        d[j] = u_prev[j] + rhs_laplace + rhs_force

    eps = 1e-12

    if bc_approx == BCApprox.TWO_POINT_SECOND and abs(bc_left.beta) > eps:
        alpha = bc_left.alpha
        beta = bc_left.beta
        g_prev = bc_left.g(time_prev)
        g_new = bc_left.g(time_new)
        # Коэффициенты для неизвестных на новом слое
        b[0] = 1.0 + sigma - (sigma * h * alpha / beta)
        c[0] = -sigma
        # Правая часть: вклад старого слоя + источники + граничные условия
        rhs_u = u_prev[0] * (1.0 - sigma + sigma * h * alpha / beta) + sigma * u_prev[1]
        rhs_f = 0.5 * tau * (f(x[0], time_prev) + f(x[0], time_new))
        rhs_bc = -(sigma * h / beta) * (g_prev + g_new)
        d[0] = rhs_u + rhs_f + rhs_bc
    elif bc_approx == BCApprox.THREE_POINT_SECOND and abs(bc_left.beta) > eps:
        alpha = bc_left.alpha
        beta = bc_left.beta
        g_prev = bc_left.g(time_prev)
        g_new = bc_left.g(time_new)
        theta = 0.5
        # Левая часть
        b[0] = -2.0 * sigma * theta
        c[0] = 2.0 * sigma * theta - 1.0
        # Правая часть
        laplace_prev = u_prev[0] - 2.0 * u_prev[1] + u_prev[2]
        d[0] = 2.0 * sigma * theta * h * g_new / beta - (u_prev[1] + (1.0 - theta) * sigma * laplace_prev)
    else:
        apply_left_bc_to_system(
            a, b, c, d, bc_left, bc_approx, sigma, h, time_new, scheme="cn"
        )

    if bc_approx == BCApprox.TWO_POINT_SECOND and abs(bc_right.beta) > eps:
        alpha = bc_right.alpha
        beta = bc_right.beta
        g_prev = bc_right.g(time_prev)
        g_new = bc_right.g(time_new)
        # Коэффициенты для неизвестных на новом слое
        a[-1] = -sigma
        b[-1] = 1.0 + sigma + (sigma * h * alpha / beta)
        # Правая часть: вклад старого слоя + источники + граничные условия
        rhs_u = u_prev[-1] * (1.0 - sigma - sigma * h * alpha / beta) + sigma * u_prev[-2]
        rhs_f = 0.5 * tau * (f(x[-1], time_prev) + f(x[-1], time_new))
        rhs_bc = (sigma * h / beta) * (g_prev + g_new)
        d[-1] = rhs_u + rhs_f + rhs_bc
    elif bc_approx == BCApprox.THREE_POINT_SECOND and abs(bc_right.beta) > eps:
        alpha = bc_right.alpha
        beta = bc_right.beta
        g_prev = bc_right.g(time_prev)
        g_new = bc_right.g(time_new)
        theta = 0.5
        # Левая часть
        a[-1] = 1.0 - 2.0 * sigma * theta
        b[-1] = 2.0 * sigma * theta
        # Правая часть
        laplace_prev = u_prev[-3] - 2.0 * u_prev[-2] + u_prev[-1]
        d[-1] = 2.0 * sigma * theta * h * g_new / beta + (u_prev[-2] + (1.0 - theta) * sigma * laplace_prev)
    else:
        apply_right_bc_to_system(
            a, b, c, d, bc_right, bc_approx, sigma, h, time_new, scheme="cn"
        )

    return a, b, c, d


def resolve_left_boundary(
    u_layer: np.ndarray,
    bc: BoundaryCondition,
    bc_approx: BCApprox,
    h: float,
    time_value: float,
) -> float:
    """Возвращает значение в левом узле, удовлетворяющем условию alpha*u + beta*u_x = g."""

    alpha, beta = bc.alpha, bc.beta
    g_val = bc.g(time_value)

    if abs(beta) < 1e-12:
        return g_val / alpha

    if bc_approx == BCApprox.TWO_POINT_FIRST:
        return (g_val - (beta / h) * u_layer[1]) / (alpha - beta / h)
    if bc_approx == BCApprox.THREE_POINT_SECOND:
        denom = alpha - 3.0 * beta / (2.0 * h)
        correction = (beta / (2.0 * h)) * (4.0 * u_layer[1] - u_layer[2])
        return (g_val - correction) / denom
    if bc_approx == BCApprox.TWO_POINT_SECOND:
        denom = alpha - beta / (2.0 * h)
        correction = (beta / (2.0 * h)) * u_layer[1]
        return (g_val - correction) / denom

    raise ValueError(f"Неизвестная аппроксимация ГУ: {bc_approx}")


def resolve_right_boundary(
    u_layer: np.ndarray,
    bc: BoundaryCondition,
    bc_approx: BCApprox,
    h: float,
    time_value: float,
) -> float:
    """Возвращает значение в правом узле."""

    alpha, beta = bc.alpha, bc.beta
    g_val = bc.g(time_value)

    if abs(beta) < 1e-12:
        return g_val / alpha

    if bc_approx == BCApprox.TWO_POINT_FIRST:
        return (g_val + (beta / h) * u_layer[-2]) / (alpha + beta / h)
    if bc_approx == BCApprox.THREE_POINT_SECOND:
        denom = alpha + 3.0 * beta / (2.0 * h)
        correction = (beta / (2.0 * h)) * (4.0 * u_layer[-2] - u_layer[-3])
        return (g_val + correction) / denom
    if bc_approx == BCApprox.TWO_POINT_SECOND:
        denom = alpha + beta / (2.0 * h)
        correction = -(beta / (2.0 * h)) * u_layer[-2]
        return (g_val - correction) / denom

    raise ValueError(f"Неизвестная аппроксимация ГУ: {bc_approx}")


def apply_left_bc_to_system(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    bc: BoundaryCondition,
    bc_approx: BCApprox,
    _sigma: float,
    h: float,
    time_value: float,
    scheme: str = "implicit",
) -> None:
    """Записывает левое граничное условие в первую строку СЛАУ."""

    alpha, beta = bc.alpha, bc.beta
    g_val = bc.g(time_value)

    if abs(beta) < 1e-12:
        b[0] = alpha
        c[0] = 0.0
        d[0] = g_val
        return

    if bc_approx == BCApprox.TWO_POINT_FIRST:
        b[0] = alpha - beta / h
        c[0] = beta / h
        d[0] = g_val
        return

    if bc_approx == BCApprox.THREE_POINT_SECOND:
        b[0] = alpha - 3.0 * beta / (2.0 * h)
        c[0] = 2.0 * beta / h
        d[0] = g_val
        return

    if bc_approx == BCApprox.TWO_POINT_SECOND:
        b[0] = alpha - beta / (2.0 * h)
        c[0] = beta / (2.0 * h)
        d[0] = g_val
        return

    raise ValueError(f"Неизвестная аппроксимация ГУ: {bc_approx}")


def apply_right_bc_to_system(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    bc: BoundaryCondition,
    bc_approx: BCApprox,
    _sigma: float,
    h: float,
    time_value: float,
    scheme: str = "implicit",
) -> None:
    """Записывает правое граничное условие в последнюю строку СЛАУ."""

    alpha, beta = bc.alpha, bc.beta
    g_val = bc.g(time_value)

    if abs(beta) < 1e-12:
        b[-1] = alpha
        a[-1] = 0.0
        d[-1] = g_val
        return

    if bc_approx == BCApprox.TWO_POINT_FIRST:
        a[-1] = -beta / h
        b[-1] = alpha + beta / h
        d[-1] = g_val
        return

    if bc_approx == BCApprox.THREE_POINT_SECOND:
        a[-1] = -2.0 * beta / h
        b[-1] = alpha + 3.0 * beta / (2.0 * h)
        d[-1] = g_val
        return

    if bc_approx == BCApprox.TWO_POINT_SECOND:
        a[-1] = -beta / (2.0 * h)
        b[-1] = alpha + beta / (2.0 * h)
        d[-1] = g_val
        return

    raise ValueError(f"Неизвестная аппроксимация ГУ: {bc_approx}")


def solve_tridiagonal(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Метод прогонки"""

    n = len(b)
    cp = np.zeros(n)
    dp = np.zeros(n)

    cp[0] = c[0] / b[0] if n > 1 else 0.0
    dp[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / denom if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x


def compute_errors(
    u: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    u_exact: ScalarFunc,
    time_indices: List[int],
) -> Dict[float, Dict[str, float]]:
    """Находит max и L2 погрешности в выбранные моменты времени."""

    errors: Dict[float, Dict[str, float]] = {}
    h = x[1] - x[0]
    for idx in time_indices:
        u_exact_vals = u_exact(x, t[idx])
        diff = np.abs(u[idx, :] - u_exact_vals)
        errors[t[idx]] = {
            "max": float(np.max(diff)),
            "l2": float(np.sqrt(h * np.sum((diff) ** 2))),
        }
    return errors


def print_error_table(errors: Dict[float, Dict[str, float]]) -> None:
    """Красиво печатает словарь ошибок."""

    print("      t       max_error      l2_error")
    print("--------------------------------------")
    for t_value, err in errors.items():
        print(f"{t_value:8.4f}   {err['max']:10.3e}   {err['l2']:10.3e}")


def plot_3d_surfaces(
    x: np.ndarray,
    t: np.ndarray,
    u_numerical: np.ndarray,
    u_exact: ScalarFunc,
    scheme: Scheme,
    bc_approx: BCApprox,
) -> None:
    """Создаёт 3D графики для аналитического и численного решений."""
    
    # Создаём сетку для 3D графика
    X, T = np.meshgrid(x, t)
    U_exact = u_exact(X, T)
    
    # Создаём фигуру с двумя 3D графиками
    fig = plt.figure(figsize=(16, 6))
    
    # График аналитического решения
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, T, U_exact, cmap='viridis', alpha=0.9, edgecolor='none')
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.set_zlabel('u(x,t)')
    ax1.set_title(f'Аналитическое решение\nu(x,t) = sin(t)·cos(x)')
    ax1.view_init(elev=25, azim=45)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    
    # График численного решения
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, T, u_numerical, cmap='plasma', alpha=0.9, edgecolor='none')
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_zlabel('u(x,t)')
    ax2.set_title(f'Численное решение\n{scheme.value}, {bc_approx.value}')
    ax2.view_init(elev=25, azim=45)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.savefig(f"3d_surface_{scheme.value}_{bc_approx.value}.png", dpi=150, bbox_inches='tight')
    plt.close()


def run_experiment() -> None:
    """Основной сценарий работы программы."""

    L = np.pi / 2
    T = 0.5
    h = 0.05
    tau = 0.00025

    def f(x_val: np.ndarray | float, t_val: np.ndarray | float):
        return np.cos(x_val) * (np.cos(t_val) + np.sin(t_val))

    def u_initial(x_val: np.ndarray) -> np.ndarray:
        return np.zeros_like(x_val)

    def u_exact(x_val: np.ndarray | float, t_val: np.ndarray | float):
        return np.sin(t_val) * np.cos(x_val)

    bc_left = BoundaryCondition(alpha=1.0, beta=0.0, g=lambda t_val: np.sin(t_val))
    bc_right = BoundaryCondition(alpha=0.0, beta=1.0, g=lambda t_val: -np.sin(t_val))

    x, t_grid = build_grid(L, T, h, tau)

    schemes = {
        Scheme.EXPLICIT: explicit_scheme,
        Scheme.IMPLICIT: implicit_scheme,
        Scheme.CRANK_NICOLSON: crank_nicolson_scheme,
    }
    bc_variants = [
        BCApprox.TWO_POINT_FIRST,
        BCApprox.THREE_POINT_SECOND,
        BCApprox.TWO_POINT_SECOND,
    ]

    time_indices = [0, len(t_grid) // 2, len(t_grid) - 1]

    results: Dict[Scheme, Dict[BCApprox, np.ndarray]] = {}

    for scheme, solver in schemes.items():
        print("=" * 60)
        print(f"Схема: {scheme.value}")
        results[scheme] = {}

        for bc_variant in bc_variants:
            print(f"  Аппроксимация границы: {bc_variant.value}")
            u_num = solver(x, t_grid, h, tau, f, u_initial, bc_left, bc_right, bc_variant)
            results[scheme][bc_variant] = u_num
            errors = compute_errors(u_num, x, t_grid, u_exact, time_indices)
            print_error_table(errors)
            print()

    final_idx = len(t_grid) - 1
    for scheme in schemes:
        plt.figure(figsize=(8, 5))
        for bc_variant in bc_variants:
            u_num = results[scheme][bc_variant]
            plt.plot(x, u_num[final_idx, :], label=bc_variant.value)
        plt.plot(x, u_exact(x, t_grid[final_idx]), "k--", label="exact")
        plt.title(f"t={t_grid[final_idx]:.2f}, схема {scheme.value}")
        plt.xlabel("x")
        plt.ylabel("u")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"comparison_{scheme.value}.png", dpi=150)
        plt.close()

    # Графики для каждой схемы с разными аппроксимациями границ
    for scheme in schemes:
        scheme_results = results[scheme]
        fig, axes = plt.subplots(1, len(bc_variants), figsize=(16, 4))
        for idx, bc_variant in enumerate(bc_variants):
            ax = axes[idx]
            u_num = scheme_results[bc_variant]
            ax.plot(x, u_num[final_idx, :], label="численное", linewidth=2)
            ax.plot(x, u_exact(x, t_grid[final_idx]), "k--", label="аналитическое", linewidth=1.5)
            ax.set_title(f"{scheme.value} – {bc_variant.value}")
            ax.set_xlabel("x")
            ax.set_ylabel("u")
            ax.grid(True)
            ax.legend()
        plt.tight_layout()
        plt.savefig(f"{scheme.value}_bc_variants.png", dpi=150)
        plt.close()

    # 3D графики для всех схем и аппроксимаций
    print("=" * 60)
    print("Создание 3D графиков...")
    for scheme_key in schemes.keys():
        for bc_variant in bc_variants:
            print(f"  3D график: {scheme_key.value} - {bc_variant.value}")
            u_num = results[scheme_key][bc_variant]
            plot_3d_surfaces(x, t_grid, u_num, u_exact, scheme_key, bc_variant)

    # Анализ чувствительности к сетке для всех схем и аппроксимаций
    for scheme_key in schemes.keys():
        for bc_variant in bc_variants:
            explore_grid_sensitivity(
                L=L,
                T=T,
                f=f,
                u0=u_initial,
                u_exact=u_exact,
                bc_left=bc_left,
                bc_right=bc_right,
                scheme=scheme_key,
                bc_variant=bc_variant,
            )


def explore_grid_sensitivity(
    L: float,
    T: float,
    f: ScalarFunc,
    u0: Callable[[np.ndarray], np.ndarray],
    u_exact: ScalarFunc,
    bc_left: BoundaryCondition,
    bc_right: BoundaryCondition,
    scheme: Scheme,
    bc_variant: BCApprox,
) -> None:
    """Показывает, как погрешность зависит от h и tau."""

    h_values = [0.2, 0.1, 0.05, 0.025]
    tau_values = [0.004, 0.001, 0.00025, 0.0000625]

    print("=" * 70)
    print(f"Зависимость погрешности от h и tau ({scheme.value}, {bc_variant.value})")
    print("      h         tau      sigma    max_error      l2_error")
    print("----------------------------------------------------------")

    max_errs: List[float] = []
    l2_errs: List[float] = []
    solutions = []
    grids = []

    for h, tau in zip(h_values, tau_values):
        sigma = tau / h**2
        x, t_grid = build_grid(L, T, h, tau)

        if scheme == Scheme.EXPLICIT:
            solver = explicit_scheme
        elif scheme == Scheme.IMPLICIT:
            solver = implicit_scheme
        else:
            solver = crank_nicolson_scheme

        u_num = solver(x, t_grid, h, tau, f, u0, bc_left, bc_right, bc_variant)
        errors = compute_errors(u_num, x, t_grid, u_exact, [len(t_grid) - 1])
        key = list(errors.keys())[0]
        max_error = errors[key]["max"]
        l2_error = errors[key]["l2"]
        max_errs.append(max_error)
        l2_errs.append(l2_error)
        solutions.append(u_num)
        grids.append((x, t_grid))

        print(f"{h:7.3f}   {tau:9.5f}   {sigma:6.3f}   {max_error:10.3e}   {l2_error:10.3e}")

    # График зависимости ошибок от h
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.loglog(h_values, max_errs, "o-", label="max")
    plt.xlabel("h")
    plt.ylabel("max error")
    plt.grid(True, which="both", alpha=0.3)
    plt.title(f"{scheme.value} - {bc_variant.value}")

    plt.subplot(1, 2, 2)
    plt.loglog(h_values, l2_errs, "s-", label="l2", color="orange")
    plt.xlabel("h")
    plt.ylabel("l2 error")
    plt.grid(True, which="both", alpha=0.3)
    plt.title(f"{scheme.value} - {bc_variant.value}")

    plt.tight_layout()
    plt.savefig(f"grid_sensitivity_{scheme.value}_{bc_variant.value}.png", dpi=150)
    plt.close()

    # Графики решений для разных h и tau
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (h, tau) in enumerate(zip(h_values, tau_values)):
        ax = axes[idx]
        x, t_grid = grids[idx]
        u_num = solutions[idx]
        final_idx = len(t_grid) - 1
        sigma = tau / h**2
        
        ax.plot(x, u_num[final_idx, :], 'b-', label="численное", linewidth=2)
        ax.plot(x, u_exact(x, t_grid[final_idx]), 'r--', label="аналитическое", linewidth=1.5)
        ax.set_title(f"h={h:.3f}, τ={tau:.5f}, σ={sigma:.2f}\nmax_err={max_errs[idx]:.2e}")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Решения при разных h и τ ({scheme.value}, {bc_variant.value})")
    plt.tight_layout()
    plt.savefig(f"grid_comparison_{scheme.value}_{bc_variant.value}.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    run_experiment()

