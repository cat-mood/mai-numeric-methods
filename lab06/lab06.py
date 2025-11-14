"""Решение гиперболического уравнения явной и неявной разностными схемами.

Задача:
    u_tt = a*u_xx + b*u_x + c*u + d*u_t + f(x,t),   0 < x < L,   0 < t <= T
    
    Граничные условия (без производных):
    u(0, t) = φ₀(t)
    u(L, t) = φ₁(t)
    
    Начальные условия:
    u(x, 0) = ψ₀(x)
    u_t(x, 0) = ψ₁(x)

Аналитическое решение: u(x, t) = exp(-x) * cos(x) * cos(2*t)

Программа демонстрирует две схемы (явная "крест", неявная)
с двумя аппроксимациями второго начального условия (первый и второй порядок).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class Scheme(Enum):
    """Тип разностной схемы."""
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"


class InitApprox(Enum):
    """Порядок аппроксимации второго начального условия."""
    FIRST_ORDER = "first_order"
    SECOND_ORDER = "second_order"


ScalarFunc = Callable[[np.ndarray | float, np.ndarray | float], np.ndarray | float]


@dataclass
class BoundaryCondition:
    """Граничное условие первого рода (Дирихле): u = g(t)."""
    g: Callable[[float], float]


@dataclass
class ProblemParams:
    """Параметры задачи."""
    # Коэффициенты уравнения
    a: float  # коэффициент перед u_xx
    b: float  # коэффициент перед u_x
    c: float  # коэффициент перед u
    d: float  # коэффициент перед u_t
    
    # Функции задачи
    f: ScalarFunc  # правая часть f(x, t)
    psi0: Callable[[np.ndarray], np.ndarray]  # начальное условие u(x, 0)
    psi1: Callable[[np.ndarray], np.ndarray]  # начальное условие u_t(x, 0)
    u_exact: ScalarFunc  # аналитическое решение
    
    # Граничные условия
    bc_left: BoundaryCondition  # u(0, t) = φ₀(t)
    bc_right: BoundaryCondition  # u(L, t) = φ₁(t)
    
    # Параметры области
    L: float  # длина пространственного интервала
    T: float  # конечное время


def build_grid(L: float, T: float, h: float, tau: float) -> Tuple[np.ndarray, np.ndarray]:
    """Создаёт равномерную сетку по x и t."""
    N = int(round(L / h))
    M = int(round(T / tau))
    x = np.linspace(0.0, L, N + 1)
    t = np.linspace(0.0, T, M + 1)
    return x, t


def solve_tridiagonal(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Метод прогонки для решения трёхдиагональной СЛАУ.
    
    Args:
        a: нижняя диагональ (коэффициенты при u_{j-1})
        b: главная диагональ (коэффициенты при u_j)
        c: верхняя диагональ (коэффициенты при u_{j+1})
        d: правая часть
    
    Returns:
        Решение СЛАУ
    """
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


def apply_initial_first_order(
    u: np.ndarray,
    x: np.ndarray,
    tau: float,
    params: ProblemParams,
) -> None:
    """Аппроксимация второго начального условия с первым порядком точности.
    
    Использует разложение Тейлора: u(x, τ) ≈ u(x, 0) + τ * u_t(x, 0)
    """
    for j in range(len(x)):
        u[1, j] = params.psi0(x)[j] + tau * params.psi1(x)[j]


def apply_initial_second_order(
    u: np.ndarray,
    x: np.ndarray,
    h: float,
    tau: float,
    params: ProblemParams,
) -> None:
    """Аппроксимация второго начального условия со вторым порядком точности.
    
    Использует разложение Тейлора с учётом PDE:
    u(x, τ) ≈ u(x, 0) + τ * u_t(x, 0) + (τ²/2) * u_tt(x, 0)
    
    Из уравнения: u_tt = a*u_xx + b*u_x + c*u + d*u_t + f(x,t)
    """
    N = len(x) - 1
    psi0_vals = params.psi0(x)
    psi1_vals = params.psi1(x)
    
    for j in range(1, N):
        # Вычисляем производные начального условия
        u_xx = (psi0_vals[j + 1] - 2.0 * psi0_vals[j] + psi0_vals[j - 1]) / h**2
        u_x = (psi0_vals[j + 1] - psi0_vals[j - 1]) / (2.0 * h)
        
        # Из PDE находим u_tt(x, 0)
        u_tt = (params.a * u_xx + params.b * u_x + params.c * psi0_vals[j] + 
                params.d * psi1_vals[j] + params.f(x[j], 0.0))
        
        # Разложение Тейлора второго порядка
        u[1, j] = psi0_vals[j] + tau * psi1_vals[j] + 0.5 * tau**2 * u_tt
    
    # Граничные точки (первый порядок или из граничных условий)
    u[1, 0] = params.bc_left.g(tau)
    u[1, -1] = params.bc_right.g(tau)


def explicit_scheme(
    x: np.ndarray,
    t: np.ndarray,
    h: float,
    tau: float,
    params: ProblemParams,
    init_approx: InitApprox,
) -> np.ndarray:
    """Явная схема "крест" для гиперболического уравнения.
    
    Использует трёхслойный шаблон (слои k-1, k, k+1).
    Формула из other_lab06.cpp (строки 207-208):
    
    u[k+1][j] = ((sigma + mu) * u[k][j+1] + (-2*sigma + 2 + c*tau²) * u[k][j] 
                 + (sigma - mu) * u[k][j-1] + (-1 + d*tau/2) * u[k-1][j] 
                 + tau² * f(j*h, k*tau)) / (1 + d*tau/2)
    
    где sigma = a*tau²/h², mu = b*tau²/(2h)
    """
    M = len(t) - 1
    N = len(x) - 1
    u = np.zeros((M + 1, N + 1))
    
    # Начальное условие u(x, 0) = ψ₀(x)
    u[0, :] = params.psi0(x)
    
    # Второе начальное условие (первый или второй слой по времени)
    if init_approx == InitApprox.FIRST_ORDER:
        apply_initial_first_order(u, x, tau, params)
    else:
        apply_initial_second_order(u, x, h, tau, params)
    
    # Коэффициенты схемы
    sigma = params.a * tau**2 / h**2
    mu = params.b * tau**2 / (2.0 * h)
    
    # Основной цикл по времени
    for k in range(1, M):
        time_prev = t[k]
        time_new = t[k + 1]
        
        # Граничные условия
        u[k + 1, 0] = params.bc_left.g(time_new)
        u[k + 1, -1] = params.bc_right.g(time_new)
        
        # Внутренние узлы (явная схема "крест")
        for j in range(1, N):
            numerator = (
                (sigma + mu) * u[k, j + 1]
                + (-2.0 * sigma + 2.0 + params.c * tau**2) * u[k, j]
                + (sigma - mu) * u[k, j - 1]
                + (-1.0 + params.d * tau / 2.0) * u[k - 1, j]
                + tau**2 * params.f(x[j], time_prev)
            )
            denominator = 1.0 + params.d * tau / 2.0
            u[k + 1, j] = numerator / denominator
    
    return u


def implicit_scheme(
    x: np.ndarray,
    t: np.ndarray,
    h: float,
    tau: float,
    params: ProblemParams,
    init_approx: InitApprox,
) -> np.ndarray:
    """Неявная схема для гиперболического уравнения.
    
    Реализация из other_lab06.cpp (строки 277-314).
    На каждом временном слое решается трёхдиагональная СЛАУ.
    """
    M = len(t) - 1
    N = len(x) - 1
    n = N + 1
    u = np.zeros((M + 1, n))
    
    # Начальное условие u(x, 0) = ψ₀(x)
    u[0, :] = params.psi0(x)
    
    # Второе начальное условие
    if init_approx == InitApprox.FIRST_ORDER:
        apply_initial_first_order(u, x, tau, params)
    else:
        apply_initial_second_order(u, x, h, tau, params)
    
    # Коэффициенты для внутренних узлов
    a_coeff = params.a / h**2
    b_coeff = params.b / (2.0 * h)
    t_coeff = 1.0 / tau**2
    d_coeff = params.d / (2.0 * tau)
    
    # Коэффициенты трёхдиагональной матрицы (не меняются со временем)
    lower = np.zeros(n)
    main = np.zeros(n)
    upper = np.zeros(n)
    
    # Граничные условия в матрице
    lower[0] = 0.0
    main[0] = 1.0
    upper[0] = 0.0
    
    # Внутренние узлы
    for j in range(1, N):
        lower[j] = b_coeff - a_coeff
        main[j] = t_coeff + d_coeff + 2.0 * a_coeff
        upper[j] = -b_coeff - a_coeff
    
    # Правая граница
    lower[-1] = 0.0
    main[-1] = 1.0
    upper[-1] = 0.0
    
    # Основной цикл по времени
    for k in range(1, M):
        time_prev = t[k]
        time_new = t[k + 1]
        
        # Правая часть СЛАУ
        rhs = np.zeros(n)
        
        # Граничные условия
        rhs[0] = params.bc_left.g(time_new)
        rhs[-1] = params.bc_right.g(time_new)
        
        # Внутренние узлы
        for j in range(1, N):
            rhs[j] = (
                u[k - 1, j] * (-t_coeff + d_coeff)
                + u[k, j] * (2.0 * t_coeff + params.c)
                + params.f(x[j], time_prev)
            )
        
        # Решаем СЛАУ методом прогонки
        u[k + 1, :] = solve_tridiagonal(lower, main, upper, rhs)
    
    return u


def plot_3d_surfaces(
    x: np.ndarray,
    t: np.ndarray,
    u_numerical: np.ndarray,
    u_exact: ScalarFunc,
    scheme: Scheme,
    init_approx: InitApprox,
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
    ax1.set_title('Аналитическое решение')
    ax1.view_init(elev=25, azim=45)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    
    # График численного решения
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, T, u_numerical, cmap='plasma', alpha=0.9, edgecolor='none')
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_zlabel('u(x,t)')
    ax2.set_title(f'Численное решение\n{scheme.value}, {init_approx.value}')
    ax2.view_init(elev=25, azim=45)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.savefig(f"3d_surface_{scheme.value}_{init_approx.value}.png", dpi=150, bbox_inches='tight')
    plt.close()


def explore_grid_sensitivity(
    params: ProblemParams,
    scheme: Scheme,
    init_approx: InitApprox,
) -> None:
    """Показывает, как погрешность зависит от h и tau."""
    # Варианты параметров сетки
    h_values = [0.2, 0.1, 0.05, 0.025]
    tau_values = [0.004, 0.001, 0.00025, 0.0000625]

    print("=" * 70)
    print(f"Зависимость погрешности от h и tau ({scheme.value}, {init_approx.value})")
    print("      h         tau        max_error      l2_error")
    print("---------------------------------------------------")

    max_errs: List[float] = []
    l2_errs: List[float] = []
    solutions = []
    grids = []

    for h, tau in zip(h_values, tau_values):
        x, t_grid = build_grid(params.L, params.T, h, tau)

        if scheme == Scheme.EXPLICIT:
            solver = explicit_scheme
        else:
            solver = implicit_scheme

        u_num = solver(x, t_grid, h, tau, params, init_approx)
        errors = compute_errors(u_num, x, t_grid, params.u_exact, [len(t_grid) - 1])
        key = list(errors.keys())[0]
        max_error = errors[key]["max"]
        l2_error = errors[key]["l2"]
        max_errs.append(max_error)
        l2_errs.append(l2_error)
        solutions.append(u_num)
        grids.append((x, t_grid))

        print(f"{h:7.3f}   {tau:9.6f}   {max_error:10.3e}   {l2_error:10.3e}")

    # График зависимости ошибок от h
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.loglog(h_values, max_errs, "o-", label="max")
    plt.xlabel("h")
    plt.ylabel("max error")
    plt.grid(True, which="both", alpha=0.3)
    plt.title(f"{scheme.value} - {init_approx.value}")

    plt.subplot(1, 2, 2)
    plt.loglog(h_values, l2_errs, "s-", label="l2", color="orange")
    plt.xlabel("h")
    plt.ylabel("l2 error")
    plt.grid(True, which="both", alpha=0.3)
    plt.title(f"{scheme.value} - {init_approx.value}")

    plt.tight_layout()
    plt.savefig(f"grid_sensitivity_{scheme.value}_{init_approx.value}.png", dpi=150)
    plt.close()

    # Графики решений для разных h и tau
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (h, tau) in enumerate(zip(h_values, tau_values)):
        ax = axes[idx]
        x, t_grid = grids[idx]
        u_num = solutions[idx]
        final_idx = len(t_grid) - 1
        
        ax.plot(x, u_num[final_idx, :], 'b-', label="численное", linewidth=2)
        ax.plot(x, params.u_exact(x, t_grid[final_idx]), 'r--', label="аналитическое", linewidth=1.5)
        ax.set_title(f"h={h:.3f}, τ={tau:.6f}\nmax_err={max_errs[idx]:.2e}")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Решения при разных h и τ ({scheme.value}, {init_approx.value})")
    plt.tight_layout()
    plt.savefig(f"grid_comparison_{scheme.value}_{init_approx.value}.png", dpi=150)
    plt.close()


def run_experiment() -> None:
    """Основной сценарий работы программы."""
    
    # ========================================================================
    # ПАРАМЕТРЫ ЗАДАЧИ
    # ========================================================================
    
    # Коэффициенты уравнения u_tt = a*u_xx + b*u_x + c*u + d*u_t + f(x,t)
    a = 1.0
    b = 2.0
    c = -2.0
    d = 0.0
    
    # Правая часть уравнения
    def f(x_val: np.ndarray | float, t_val: np.ndarray | float):
        return 0 * x_val  # возвращает массив нулей той же формы, что x_val
    
    # Начальные условия
    def psi0(x_val: np.ndarray) -> np.ndarray:
        """u(x, 0) = ψ₀(x)"""
        return np.exp(-x_val) * np.cos(x_val)
    
    def psi1(x_val: np.ndarray) -> np.ndarray:
        """u_t(x, 0) = ψ₁(x)"""
        return 0 * x_val  # возвращает массив нулей той же формы, что x_val
    
    # Граничные условия
    def phi0(t_val: float) -> float:
        """u(0, t) = φ₀(t)"""
        return np.cos(2 * t_val)
    
    def phi1(t_val: float) -> float:
        """u(L, t) = φ₁(t)"""
        return 0.0
    
    # Аналитическое решение
    def u_exact(x_val: np.ndarray | float, t_val: np.ndarray | float):
        """Точное решение u(x, t)"""
        return np.exp(-x_val) * np.cos(x_val) * np.cos(2 * t_val)
    
    # Параметры области
    L = np.pi / 2
    T = 3
    
    # Параметры сетки
    h = 0.05
    tau = 0.0005
    
    # ========================================================================
    # КОНЕЦ ПАРАМЕТРОВ
    # ========================================================================
    
    bc_left = BoundaryCondition(g=phi0)
    bc_right = BoundaryCondition(g=phi1)
    
    params = ProblemParams(
        a=a, b=b, c=c, d=d,
        f=f,
        psi0=psi0,
        psi1=psi1,
        u_exact=u_exact,
        bc_left=bc_left,
        bc_right=bc_right,
        L=L,
        T=T,
    )
    
    x, t_grid = build_grid(L, T, h, tau)
    
    schemes = {
        Scheme.EXPLICIT: explicit_scheme,
        Scheme.IMPLICIT: implicit_scheme,
    }
    
    init_variants = [
        InitApprox.FIRST_ORDER,
        InitApprox.SECOND_ORDER,
    ]
    
    time_indices = [0, len(t_grid) // 2, len(t_grid) - 1]
    
    results: Dict[Scheme, Dict[InitApprox, np.ndarray]] = {}
    
    for scheme, solver in schemes.items():
        print("=" * 60)
        print(f"Схема: {scheme.value}")
        results[scheme] = {}
        
        for init_variant in init_variants:
            print(f"  Аппроксимация начального условия: {init_variant.value}")
            u_num = solver(x, t_grid, h, tau, params, init_variant)
            results[scheme][init_variant] = u_num
            errors = compute_errors(u_num, x, t_grid, params.u_exact, time_indices)
            print_error_table(errors)
            print()
    
    # Графики для каждой схемы с разными аппроксимациями начальных условий
    final_idx = len(t_grid) - 1
    for scheme in schemes:
        plt.figure(figsize=(8, 5))
        for init_variant in init_variants:
            u_num = results[scheme][init_variant]
            plt.plot(x, u_num[final_idx, :], label=init_variant.value)
        plt.plot(x, params.u_exact(x, t_grid[final_idx]), "k--", label="exact")
        plt.title(f"t={t_grid[final_idx]:.2f}, схема {scheme.value}")
        plt.xlabel("x")
        plt.ylabel("u")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"comparison_{scheme.value}.png", dpi=150)
        plt.close()
    
    # Графики для каждой схемы с разными аппроксимациями
    for scheme in schemes:
        scheme_results = results[scheme]
        fig, axes = plt.subplots(1, len(init_variants), figsize=(14, 4))
        if len(init_variants) == 1:
            axes = [axes]
        for idx, init_variant in enumerate(init_variants):
            ax = axes[idx]
            u_num = scheme_results[init_variant]
            ax.plot(x, u_num[final_idx, :], label="численное", linewidth=2)
            ax.plot(x, params.u_exact(x, t_grid[final_idx]), "k--", label="аналитическое", linewidth=1.5)
            ax.set_title(f"{scheme.value} – {init_variant.value}")
            ax.set_xlabel("x")
            ax.set_ylabel("u")
            ax.grid(True)
            ax.legend()
        plt.tight_layout()
        plt.savefig(f"{scheme.value}_init_variants.png", dpi=150)
        plt.close()
    
    # 3D графики для всех схем и аппроксимаций
    print("=" * 60)
    print("Создание 3D графиков...")
    for scheme_key in schemes.keys():
        for init_variant in init_variants:
            print(f"  3D график: {scheme_key.value} - {init_variant.value}")
            u_num = results[scheme_key][init_variant]
            plot_3d_surfaces(x, t_grid, u_num, params.u_exact, scheme_key, init_variant)
    
    # Анализ чувствительности к сетке для всех схем и аппроксимаций
    print("=" * 60)
    print("Анализ чувствительности к параметрам сетки...")
    for scheme_key in schemes.keys():
        for init_variant in init_variants:
            explore_grid_sensitivity(params, scheme_key, init_variant)


if __name__ == "__main__":
    run_experiment()

