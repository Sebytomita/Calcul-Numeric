import math
import random

def generate_spd_matrix(n, seed=42):
    random.seed(seed)
    B = [[random.uniform(1, 5) for _ in range(n)] for _ in range(n)]
    
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += B[i][k] * B[j][k]
            A[i][j] = s
    return A

def generate_vector_b(n, seed=99):
    random.seed(seed)
    return [random.uniform(1, 10) for _ in range(n)]

def copy_matrix(A):
    return [row[:] for row in A]

def print_matrix(A, name="A", max_size=6):
    n = len(A)
    display = min(n, max_size)
    print(f"\nMatricea {name} ({n}x{n}){' [primele ' + str(display) + ' linii/col]' if n > max_size else ''}:")
    for i in range(display):
        row_str = "  ["
        for j in range(display):
            row_str += f"{A[i][j]:10.4f}"
        if n > max_size:
            row_str += "  ..."
        row_str += " ]"
        print(row_str)
    if n > max_size:
        print("  ...")

def print_vector(v, name="v", max_size=8):
    n = len(v)
    display = min(n, max_size)
    vals = "  ".join(f"{v[i]:.6f}" for i in range(display))
    suffix = "  ..." if n > max_size else ""
    print(f"{name} = [{vals}{suffix}]")

def mat_vec_Ainit(A_chol, x):
    n = len(A_chol)
    result = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(n):
            if j >= i:
                aij = A_chol[i][j]
            else:
                aij = A_chol[j][i]
            s += aij * x[j]
        result[i] = s
    return result

def norm2(v):
    return math.sqrt(sum(x * x for x in v))

def vec_diff(a, b):
    return [a[i] - b[i] for i in range(len(a))]

def check_symmetry(A, eps=1e-10):
    """Verifică dacă matricea A este simetrică (pentru debugging)."""
    n = len(A)
    for i in range(n):
        for j in range(i):
            if abs(A[i][j] - A[j][i]) > eps:
                return False, (i, j, A[i][j], A[j][i])
    return True, None

def cholesky_ldlt(A, eps):
    n = len(A)
    d = [0.0] * n

    for p in range(n):
        dp = A[p][p]
        for k in range(p):
            dp -= d[k] * A[p][k] * A[p][k]
        
        if abs(dp) <= eps:
            raise ValueError(f"Descompunerea LDLT a eșuat la pasul p={p}: dp={dp} ≈ 0")
        
        d[p] = dp

        for i in range(p + 1, n):
            s = A[p][i]
            for k in range(p):
                s -= d[k] * A[i][k] * A[p][k]
            A[i][p] = s / dp

    return d

def forward_substitution_L(A, b):
    n = len(b)
    z = [0.0] * n
    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= A[i][j] * z[j]
        z[i] = s
    return z

def solve_diagonal(d, z, eps):
    n = len(z)
    y = [0.0] * n
    for i in range(n):
        if abs(d[i]) <= eps:
            raise ValueError(f"Element diagonal zero la i={i}")
        y[i] = z[i] / d[i]
    return y

def backward_substitution_Lt(A, y):
    n = len(y)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= A[j][i] * x[j]
        x[i] = s
    return x

def solve_cholesky(A_chol, d, b, eps):
    z = forward_substitution_L(A_chol, b)
    y = solve_diagonal(d, z, eps)
    x = backward_substitution_Lt(A_chol, y)
    return x

def compute_determinant(d):
    result = 1.0
    for di in d:
        result *= di
    return result

def solve_with_library(A_init, b):
    try:
        import numpy as np
        from scipy import linalg
        
        An = np.array(A_init)
        bn = np.array(b)
        
        P, L, U = linalg.lu(An)
        print("\n--- Descompunere LU (scipy) ---")
        print(f"  Shape L: {L.shape}, Shape U: {U.shape}")
        
        x_lib = linalg.solve(An, bn)
        return x_lib.tolist()
    except ImportError:
        print("\n[ATENȚIE] scipy/numpy nu sunt instalate. Folosim Gauss pentru x_lib.")
        return gauss_elimination(A_init, b)

def gauss_elimination(A_orig, b_orig):
    n = len(b_orig)
    A = [row[:] + [b_orig[i]] for i, row in enumerate(A_orig)]
    
    for col in range(n):
        max_row = max(range(col, n), key=lambda r: abs(A[r][col]))
        A[col], A[max_row] = A[max_row], A[col]
        
        if abs(A[col][col]) < 1e-12:
            raise ValueError(f"Matrice singulară la coloana {col}")
        
        for row in range(col + 1, n):
            factor = A[row][col] / A[col][col]
            for j in range(col, n + 1):
                A[row][j] -= factor * A[col][j]
    
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = A[i][n]
        for j in range(i + 1, n):
            x[i] -= A[i][j] * x[j]
        x[i] /= A[i][i]
    return x

def main():
    print("=" * 60)
    print("TEMA 2 - Descompunerea Cholesky (LDLT)")
    print("=" * 60)

    try:
        n = int(input("\nIntroduceți dimensiunea n a matricei: "))
        t = int(input("Introduceți t (epsilon = 10^(-t), t=5..10): "))
    except ValueError:
        print("Input invalid. Se folosesc valorile implicite n=4, t=8.")
        n, t = 4, 8

    eps = 10.0 ** (-t)
    print(f"\nn = {n}, epsilon = 10^(-{t}) = {eps}")

    A_init = generate_spd_matrix(n)
    b = generate_vector_b(n)

    print_matrix(A_init, "A_init")
    print_vector(b, "b")

    print("\n" + "=" * 60)
    print("1. SOLUȚIE CU BIBLIOTECA (LU)")
    print("=" * 60)
    x_lib = solve_with_library(A_init, b)
    print_vector(x_lib, "x_lib")

    print("\n" + "=" * 60)
    print("2. DESCOMPUNEREA CHOLESKY (LDLT)")
    print("=" * 60)

    A = copy_matrix(A_init)

    try:
        d = cholesky_ldlt(A, eps)
    except ValueError as e:
        print(f"EROARE: {e}")
        return

    print("\nMatricea A după descompunere:")
    print("  (triunghi superior = A_init, triunghi strict inferior = L)")
    print_matrix(A, "A_chol")
    print_vector(d, "d (diagonala D)")

    if n == 3:
        print("\n[Verificare exemplu din suport]")
        A_ex = [[1, 2.5, 3], [2.5, 8.25, 15.5], [3, 15.5, 43]]
        d_ex = cholesky_ldlt(A_ex, eps)
        print_matrix(A_ex, "A_exemplu_dupa_cholesky")
        print_vector(d_ex, "d_exemplu")
        print("Rezultat așteptat: L[2][0]=2.5, L[2][1]=4, d=[1,2,2]")

    print("\n" + "=" * 60)
    print("3. DETERMINANTUL")
    print("=" * 60)
    det_A = compute_determinant(d)
    print(f"det(A) = prod(d_i) = {det_A:.6f}")
    
    try:
        import numpy as np
        det_np = np.linalg.det(np.array(A_init))
        print(f"det(A) numpy       = {det_np:.6f}")
        print(f"Diferență relativă = {abs(det_A - det_np) / abs(det_np):.2e}")
    except ImportError:
        pass

    print("\n" + "=" * 60)
    print("4. REZOLVARE SISTEM Ax=b CU CHOLESKY")
    print("=" * 60)
    x_chol = solve_cholesky(A, d, b, eps)
    print_vector(x_chol, "x_Chol")

    print("\n" + "=" * 60)
    print("5. VERIFICARE SOLUȚIE")
    print("=" * 60)

    Ax = mat_vec_Ainit(A, x_chol)
    residual = vec_diff(Ax, b)
    norm_residual = norm2(residual)
    print(f"||A_init * x_Chol - b||_2  = {norm_residual:.2e}  (target: < 1e-8)")

    diff_x = vec_diff(x_chol, x_lib)
    norm_diff = norm2(diff_x)
    print(f"||x_Chol - x_lib||_2       = {norm_diff:.2e}  (target: < 1e-9)")

    print()
    ok1 = norm_residual < 1e-8
    ok2 = norm_diff < 1e-9
    print(f"Norma reziduală OK  (< 1e-8): {'✓ DA' if ok1 else '✗ NU'}")
    print(f"Diferența față de lib OK: {'✓ DA' if ok2 else '✗ NU'}")

    print("\n" + "=" * 60)
    print("6. TEST MATRICE MARE (n=100)")
    print("=" * 60)
    import time
    n_big = 100
    A_big = generate_spd_matrix(n_big, seed=7)
    b_big = generate_vector_b(n_big, seed=13)
    A_big_copy = copy_matrix(A_big)

    t0 = time.perf_counter()
    d_big = cholesky_ldlt(A_big_copy, eps)
    x_big = solve_cholesky(A_big_copy, d_big, b_big, eps)
    t1 = time.perf_counter()

    Ax_big = mat_vec_Ainit(A_big_copy, x_big)
    res_big = norm2(vec_diff(Ax_big, b_big))
    print(f"n=100: timp={t1-t0:.4f}s, ||Ax-b||_2={res_big:.2e}")
    print(f"Soluție OK: {'✓' if res_big < 1e-6 else '✗'}")


if __name__ == "__main__":
    main()