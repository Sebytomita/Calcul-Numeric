import math
import random

def f(x):
    return x**4 - 12*x**3 + 30*x**2 + 12

def df(x):
    return 4*x**3 - 36*x**2 + 60*x

def solve_linear(A_in, b_in):
    n = len(b_in)
    A = [row[:] + [b_in[i]] for i, row in enumerate(A_in)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(A[r][col]))
        A[col], A[pivot] = A[pivot], A[col]
        if abs(A[col][col]) < 1e-14:
            raise ValueError(f"Sistem singular la coloana {col}")
        for row in range(col + 1, n):
            f_val = A[row][col] / A[col][col]
            for j in range(col, n + 1):
                A[row][j] -= f_val * A[col][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = A[i][n]
        for j in range(i + 1, n):
            x[i] -= A[i][j] * x[j]
        x[i] /= A[i][i]
    return x

def generate_nodes(x0, xn, n, seed=42):
    random.seed(seed)
    interior = sorted([random.uniform(x0, xn) for _ in range(n - 1)])
    xs = [x0] + interior + [xn]
    ys = [f(x) for x in xs]
    return xs, ys

def horner(coeffs, x0):
    c = list(reversed(coeffs))
    d = c[0]
    for i in range(1, len(c)):
        d = c[i] + d * x0
    return d

def least_squares_poly(xs, ys, m):
    n_pts = len(xs)
    max_pow = 2 * m
    pow_x = [[xs[k]**p for p in range(max_pow + 1)] for k in range(n_pts)]
    B = [[0.0] * (m + 1) for _ in range(m + 1)]
    rhs = [0.0] * (m + 1)
    for i in range(m + 1):
        for j in range(m + 1):
            B[i][j] = sum(pow_x[k][i + j] for k in range(n_pts))
        rhs[i] = sum(ys[k] * pow_x[k][i] for k in range(n_pts))
    coeffs = solve_linear(B, rhs)
    return coeffs

def eval_poly(coeffs, x):
    return horner(coeffs, x)

def build_spline(xs, ys, da, db):
    n = len(xs) - 1
    h = [xs[i + 1] - xs[i] for i in range(n)]
    size = n + 1
    H = [[0.0] * size for _ in range(size)]
    rhs = [0.0] * size
    H[0][0] = 2 * h[0]
    H[0][1] = h[0]
    rhs[0] = 6 * ((ys[1] - ys[0]) / h[0] - da)
    for i in range(1, n):
        H[i][i - 1] = h[i - 1]
        H[i][i]     = 2 * (h[i - 1] + h[i])
        H[i][i + 1] = h[i]
        rhs[i] = 6 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1])
    H[n][n - 1] = h[n - 1]
    H[n][n]     = 2 * h[n - 1]
    rhs[n] = 6 * (db - (ys[n] - ys[n - 1]) / h[n - 1])
    A = solve_linear(H, rhs)
    b_coef = [0.0] * n
    c_coef = [0.0] * n
    for i in range(n):
        b_coef[i] = (ys[i + 1] - ys[i]) / h[i] - h[i] * (A[i + 1] - A[i]) / 6
        c_coef[i] = (xs[i + 1] * ys[i] - xs[i] * ys[i + 1]) / h[i] \
                    - h[i] * (xs[i + 1] * A[i] - xs[i] * A[i + 1]) / 6
    return A, b_coef, c_coef, h

def eval_spline(x_bar, xs, ys, A, b_coef, c_coef, h):
    n = len(xs) - 1
    i0 = 0
    for i in range(n):
        if xs[i] <= x_bar <= xs[i + 1]:
            i0 = i
            break
    hi0 = h[i0]
    val = ((x_bar - xs[i0])**3 * A[i0 + 1] / (6 * hi0)
           + (xs[i0 + 1] - x_bar)**3 * A[i0] / (6 * hi0)
           + b_coef[i0] * x_bar
           + c_coef[i0])
    return val

def plot_ascii(xs_nodes, ys_nodes, f_func, poly_coeffs, spline_data, x0, xn, x_bar):
    width, height = 70, 25
    n_pts = 200
    step = (xn - x0) / (n_pts - 1)
    x_vals = [x0 + i * step for i in range(n_pts)]
    f_vals  = [f_func(x) for x in x_vals]
    pm_vals = [eval_poly(poly_coeffs, x) for x in x_vals]
    A_sp, b_sp, c_sp, h_sp = spline_data
    sf_vals = [eval_spline(x, xs_nodes, ys_nodes, A_sp, b_sp, c_sp, h_sp) for x in x_vals]
    y_min = min(min(f_vals), min(pm_vals), min(sf_vals))
    y_max = max(max(f_vals), max(pm_vals), max(sf_vals))
    y_range = y_max - y_min or 1.0

    def to_row(y):
        return int((height - 1) * (1 - (y - y_min) / y_range))

    grid = [[' '] * width for _ in range(height)]

    def plot_curve(vals, char):
        for idx, y in enumerate(vals):
            col = int(idx * (width - 1) / (n_pts - 1))
            row = max(0, min(height - 1, to_row(y)))
            grid[row][col] = char

    plot_curve(f_vals,  'f')
    plot_curve(sf_vals, 's')
    plot_curve(pm_vals, 'p')
    xbar_col = int((x_bar - x0) / (xn - x0) * (width - 1))
    for row in range(height):
        if grid[row][xbar_col] == ' ':
            grid[row][xbar_col] = '|'
    print(f"\n  Grafic ASCII: f=f(x), p=Pm(x), s=Sf(x), |=x_bar")
    print(f"  y_max={y_max:.2f}")
    print("  " + "-" * width)
    for row in grid:
        print("  |" + "".join(row) + "|")
    print("  " + "-" * width)
    print(f"  y_min={y_min:.2f}")
    print(f"  x: [{x0:.2f} ... {xn:.2f}]")

def main():
    print("=" * 60)
    print("TEMA 6 - Aproximare polinomială și spline cubic")
    print("=" * 60)
    try:
        x0 = float(input("\nIntroduceți x0 (capătul stâng, ex. 0): "))
        xn = float(input("Introduceți xn (capătul drept, ex. 2): "))
        if x0 >= xn:
            raise ValueError("x0 trebuie < xn")
        n = int(input("Introduceți n (numărul de intervale, ex. 8): "))
        m = int(input("Introduceți m (gradul polinomului, m<6, ex. 3): "))
        x_bar = float(input(f"Introduceți x_bar ∈ ({x0}, {xn}): "))
    except (ValueError, EOFError):
        print("Input invalid. Se folosesc valorile din PDF: x0=0, xn=2, n=8, m=3, x_bar=0.5")
        x0, xn, n, m, x_bar = 0.0, 2.0, 8, 3, 0.5

    if m >= 6:
        print("m trebuie < 6. Se setează m=5.")
        m = 5

    da = df(x0)
    db = df(xn)

    print(f"\nx0={x0}, xn={xn}, n={n}, m={m}, x_bar={x_bar}")
    print(f"f(x) = x^4 - 12x^3 + 30x^2 + 12")
    print(f"da = f'({x0}) = {da:.6f}")
    print(f"db = f'({xn}) = {db:.6f}")
    print(f"f(x_bar) = f({x_bar}) = {f(x_bar):.6f}")

    xs, ys = generate_nodes(x0, xn, n)
    print(f"\nNoduri generate (n+1={n+1} puncte):")
    print("  xi: " + "  ".join(f"{x:.4f}" for x in xs))
    print("  yi: " + "  ".join(f"{y:.4f}" for y in ys))

    print("\n" + "=" * 60)
    print(f"1. APROXIMARE POLINOMIALĂ (grad m={m})")
    print("=" * 60)

    coeffs = least_squares_poly(xs, ys, m)
    print(f"\nCoeficienți Pm (a0, a1, ..., a{m}):")
    for i, c in enumerate(coeffs):
        print(f"  a{i} = {c:.8f}")

    pm_xbar = eval_poly(coeffs, x_bar)
    f_xbar  = f(x_bar)
    err_pm  = abs(pm_xbar - f_xbar)

    print(f"\nEvaluare în x_bar = {x_bar}:")
    print(f"  Pm(x_bar)        = {pm_xbar:.8f}")
    print(f"  f(x_bar)         = {f_xbar:.8f}")
    print(f"  |Pm(x_bar) - f(x_bar)| = {err_pm:.2e}")

    sum_err = sum(abs(eval_poly(coeffs, xs[i]) - ys[i]) for i in range(n + 1))
    print(f"\n  Σ|Pm(xi) - yi| = {sum_err:.6f}")

    print(f"\nComparație erori pentru diferite grade m:")
    print(f"  {'m':>4} | {'|Pm(x_bar)-f(x_bar)|':>22} | {'Σ|Pm(xi)-yi|':>15}")
    print("  " + "-" * 48)
    for m_test in range(1, 6):
        try:
            c_test = least_squares_poly(xs, ys, m_test)
            e1 = abs(eval_poly(c_test, x_bar) - f_xbar)
            e2 = sum(abs(eval_poly(c_test, xs[i]) - ys[i]) for i in range(n + 1))
            print(f"  {m_test:>4} | {e1:>22.2e} | {e2:>15.6f}")
        except Exception as e:
            print(f"  {m_test:>4} | {'EROARE':>22} | {str(e)[:15]:>15}")

    print("\n" + "=" * 60)
    print("2. SPLINE CUBIC C2")
    print("=" * 60)

    A_sp, b_sp, c_sp, h_sp = build_spline(xs, ys, da, db)

    print(f"\nConstantele Ai (i=0,...,{n}):")
    for i, a in enumerate(A_sp):
        print(f"  A{i} = {a:.8f}")

    sf_xbar = eval_spline(x_bar, xs, ys, A_sp, b_sp, c_sp, h_sp)
    err_sf  = abs(sf_xbar - f_xbar)

    print(f"\nEvaluare în x_bar = {x_bar}:")
    print(f"  Sf(x_bar)        = {sf_xbar:.8f}")
    print(f"  f(x_bar)         = {f_xbar:.8f}")
    print(f"  |Sf(x_bar) - f(x_bar)| = {err_sf:.2e}")

    print("\n" + "=" * 60)
    print("VERIFICARE EXEMPLU DIN PDF")
    print("=" * 60)
    xs_pdf = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    ys_pdf = [50.0, 47.0, -2.0, -121.0, -310.0, -545.0]
    da_pdf, db_pdf = 6.0, -244.0
    xbar_pdf = 1.5
    f_pdf = 30.3125

    A_pdf, b_pdf, c_pdf, h_pdf = build_spline(xs_pdf, ys_pdf, da_pdf, db_pdf)
    sf_pdf = eval_spline(xbar_pdf, xs_pdf, ys_pdf, A_pdf, b_pdf, c_pdf, h_pdf)
    print(f"Sf(1.5) = {sf_pdf:.6f}  (așteptat ≈ 30.25)")
    print(f"|Sf(1.5) - f(1.5)| = |{sf_pdf:.4f} - {f_pdf}| = {abs(sf_pdf - f_pdf):.4f}")

    print("\n" + "=" * 60)
    print("3. GRAFIC COMPARATIV")
    print("=" * 60)

    spline_data = (A_sp, b_sp, c_sp, h_sp)
    plot_ascii(xs, ys, f, coeffs, spline_data, x0, xn, x_bar)

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        x_plot = np.linspace(x0, xn, 300)
        y_f    = [f(x) for x in x_plot]
        y_pm   = [eval_poly(coeffs, x) for x in x_plot]
        y_sf   = [eval_spline(x, xs, ys, A_sp, b_sp, c_sp, h_sp) for x in x_plot]

        plt.figure(figsize=(10, 6))
        plt.plot(x_plot, y_f,  'b-',  linewidth=2, label='f(x) exactă')
        plt.plot(x_plot, y_pm, 'r--', linewidth=2, label=f'Pm(x) grad {m}')
        plt.plot(x_plot, y_sf, 'g-.',  linewidth=2, label='Sf(x) spline cubic')
        plt.scatter(xs, ys, color='black', zorder=5, label='noduri (xi, yi)')
        plt.axvline(x=x_bar, color='purple', linestyle=':', label=f'x_bar={x_bar}')
        plt.scatter([x_bar], [f_xbar], color='purple', zorder=6, s=100)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Aproximare f(x) = x⁴ - 12x³ + 30x² + 12\nm={m}, n={n} noduri')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('tema6_grafic.png', dpi=150)
        plt.show()
        print("\nGrafic salvat în tema6_grafic.png")
    except ImportError:
        print("\n(matplotlib indisponibil — se afișează doar graficul ASCII)")
        print("Instalați cu: pip install matplotlib")

if __name__ == "__main__":
    main()