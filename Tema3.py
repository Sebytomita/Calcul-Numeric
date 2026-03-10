import math
import random

def copy_matrix(A):
    return [row[:] for row in A]

def copy_vector(v):
    return v[:]

def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

def norm2(v):
    return math.sqrt(sum(x * x for x in v))

def vec_diff(a, b):
    return [a[i] - b[i] for i in range(len(a))]

def mat_vec(A, x):
    n = len(A)
    return [sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]

def print_matrix(A, name="A", max_size=6):
    n = len(A)
    d = min(n, max_size)
    suffix = f" [primele {d} linii/col]" if n > max_size else ""
    print(f"\nMatricea {name} ({n}x{n}){suffix}:")
    for i in range(d):
        row = "  [" + "".join(f"{A[i][j]:10.4f}" for j in range(d))
        if n > max_size:
            row += "  ..."
        print(row + " ]")
    if n > max_size:
        print("  ...")

def print_vector(v, name="v", max_size=8):
    n = len(v)
    d = min(n, max_size)
    vals = "  ".join(f"{v[i]:.6f}" for i in range(d))
    suffix = "  ..." if n > max_size else ""
    print(f"{name} = [{vals}{suffix}]")

# Generare date

def generate_matrix(n, seed=42):
    random.seed(seed)
    A = [[random.uniform(-5, 5) for _ in range(n)] for _ in range(n)]
    # pt singularitate
    for i in range(n):
        A[i][i] += n * 2.0
    return A

def generate_vector_s(n, seed=7):
    random.seed(seed)
    return [random.uniform(1, 10) for _ in range(n)]

# ex1

def compute_b(A, s):
    n = len(A)
    return [sum(s[j] * A[i][j] for j in range(n)) for i in range(n)]

# ex2
def householder_qr(A_in, b_in, eps):
    
    n = len(A_in)
    A = copy_matrix(A_in)
    b = copy_vector(b_in)
    Q_bar = identity(n)

    for r in range(n - 1):
        
        sigma = sum(A[i][r] ** 2 for i in range(r, n))
        
        if sigma <= eps:
            break  # A singulara
        
        k = math.sqrt(sigma)
        if A[r][r] > 0:
            k = -k
        
        beta = sigma - k * A[r][r]
        
        u = [0.0] * n
        u[r] = A[r][r] - k
        for i in range(r + 1, n):
            u[i] = A[i][r]
        
        for j in range(r + 1, n):
            gamma = sum(u[i] * A[i][j] for i in range(r, n)) / beta
            for i in range(r, n):
                A[i][j] -= gamma * u[i]
        
        # Transf col r a lui A
        A[r][r] = k
        for i in range(r + 1, n):
            A[i][r] = 0.0
        
        # Transf b = P_r * b
        gamma = sum(u[i] * b[i] for i in range(r, n)) / beta
        for i in range(r, n):
            b[i] -= gamma * u[i]
        
        # Transf Q_bar = P_r * Q_bar
        for j in range(n):
            gamma = sum(u[i] * Q_bar[i][j] for i in range(r, n)) / beta
            for i in range(r, n):
                Q_bar[i][j] -= gamma * u[i]
    
    return A, Q_bar, b

# ex3  

def back_substitution(R, b, eps):
    n = len(b)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(R[i][i]) <= eps:
            raise ValueError(f"Element diagonal zero la i={i}: R[i][i]={R[i][i]}")
        s = b[i]
        for j in range(i + 1, n):
            s -= R[i][j] * x[j]
        x[i] = s / R[i][i]
    return x

# ex5

def compute_inverse_householder(R, Qt, eps):
    n = len(R)
    inv_A = [[0.0] * n for _ in range(n)]
    
    for j in range(n):
        b_j = [Qt[i][j] for i in range(n)]
        x = back_substitution(R, b_j, eps)
        for i in range(n):
            inv_A[i][j] = x[i]
    
    return inv_A

def solve_library(A_init, b):
    try:
        import numpy as np
        from scipy import linalg
        An = np.array(A_init)
        bn = np.array(b)
        Q_lib, R_lib = linalg.qr(An)
        x_qr = linalg.solve(An, bn)
        inv_lib = linalg.inv(An)
        return x_qr.tolist(), inv_lib.tolist()
    except ImportError:
        # Fallback: Gauss
        return gauss_solve(A_init, b), None

def gauss_solve(A_orig, b_orig):
    n = len(b_orig)
    A = [row[:] + [b_orig[i]] for i, row in enumerate(A_orig)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(A[r][col]))
        A[col], A[pivot] = A[pivot], A[col]
        for row in range(col + 1, n):
            f = A[row][col] / A[col][col]
            for j in range(col, n + 1):
                A[row][j] -= f * A[col][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = A[i][n]
        for j in range(i + 1, n):
            x[i] -= A[i][j] * x[j]
        x[i] /= A[i][i]
    return x

def matrix_diff_norm(A, B):
    n = len(A)
    return math.sqrt(sum((A[i][j] - B[i][j])**2 for i in range(n) for j in range(n)))

def main():
    print("=" * 60)
    print("TEMA 3 - Descompunerea QR (Algoritmul Householder)")
    print("=" * 60)

    try:
        n = int(input("\nIntroduceți dimensiunea n: "))
        t = int(input("Introduceți t (epsilon = 10^(-t)): "))
    except ValueError:
        print("Input invalid. Se folosesc n=4, t=8.")
        n, t = 4, 8

    eps = 10.0 ** (-t)
    print(f"\nn = {n}, epsilon = 10^(-{t}) = {eps}")

    A_init = generate_matrix(n)
    s = generate_vector_s(n)

    # ex1
    b_init = compute_b(A_init, s)

    print_matrix(A_init, "A_init")
    print_vector(s, "s")
    print_vector(b_init, "b = A*s")

    # ex2
    print("\n" + "=" * 60)
    print("2. DESCOMPUNEREA QR (Householder)")
    print("=" * 60)

    R, Qt, Qtb = householder_qr(A_init, b_init, eps)

    for i in range(n):
        if abs(R[i][i]) <= eps:
            print(f"EROARE: Matricea A este singulară (R[{i}][{i}] ≈ 0)")
            return

    print_matrix(R, "R (matricea superior triunghiulară)")
    print_vector(Qtb, "Q^T * b")

    # ex3
    print("\n" + "=" * 60)
    print("3. REZOLVARE SISTEM Ax = b")
    print("=" * 60)

    x_householder = back_substitution(R, Qtb, eps)

    x_lib, inv_lib = solve_library(A_init, b_init)
    x_qr = x_lib

    print_vector(x_householder, "x_Householder")
    print_vector(x_qr, "x_QR (biblioteca)")

    diff_xqr_xh = norm2(vec_diff(x_qr, x_householder))
    print(f"\n||x_QR - x_Householder||_2 = {diff_xqr_xh:.2e}")

    # ex4
    print("\n" + "=" * 60)
    print("4. ERORI")
    print("=" * 60)

    Ax_h = mat_vec(A_init, x_householder)
    err1 = norm2(vec_diff(Ax_h, b_init))

    Ax_qr = mat_vec(A_init, x_qr)
    err2 = norm2(vec_diff(Ax_qr, b_init))

    norm_s = norm2(s)
    err3 = norm2(vec_diff(x_householder, s)) / norm_s

    err4 = norm2(vec_diff(x_qr, s)) / norm_s

    print(f"||A_init * x_Householder - b_init||_2 = {err1:.2e}  (target < 1e-6)")
    print(f"||A_init * x_QR - b_init||_2          = {err2:.2e}  (target < 1e-6)")
    print(f"||x_Householder - s||_2 / ||s||_2     = {err3:.2e}  (target < 1e-6)")
    print(f"||x_QR - s||_2 / ||s||_2              = {err4:.2e}  (target < 1e-6)")

    ok = all(v < 1e-6 for v in [err1, err2, err3, err4])
    print(f"\nToate erorile < 1e-6: {'✓ DA' if ok else '✗ NU'}")

    # ex5
    print("\n" + "=" * 60)
    print("5. INVERSA MATRICEI A")
    print("=" * 60)

    inv_householder = compute_inverse_householder(R, Qt, eps)

    if inv_lib is not None:
        err_inv = matrix_diff_norm(inv_householder, inv_lib)
        print(f"||A^(-1)_Householder - A^(-1)_bibl||_F = {err_inv:.2e}")
        inv_ok = err_inv < 1e-6
        print(f"Inversa OK (< 1e-6): {'✓ DA' if inv_ok else '✗ NU'}")
    else:
        print("(numpy/scipy indisponibil, nu se poate compara cu biblioteca)")

    print_matrix(inv_householder, "A^(-1)_Householder")

    # test exemplu PDF (n=3)
    print("\n" + "=" * 60)
    print(" VERIFICARE EXEMPLU DIN PDF (n=3)")
    print("=" * 60)
    A_ex = [[0, 0, 4], [1, 2, 3], [0, 1, 2]]
    s_ex = [3, 2, 1]
    b_ex = compute_b(A_ex, s_ex) #1
    print(f"b_ex calculat: {b_ex}  (așteptat: [4, 10, 4])")

    R_ex, Qt_ex, Qtb_ex = householder_qr(A_ex, b_ex, eps) #2
    x_ex = back_substitution(R_ex, Qtb_ex, eps) #3
    print(f"x_ex = {[round(v, 4) for v in x_ex]}  (așteptat: [3, 2, 1])")

    inv_ex = compute_inverse_householder(R_ex, Qt_ex, eps) #5
    print("A^(-1) exemplu (așteptat [[0.25,1,-2],[-0.5,0,1],[0.25,0,0]]):")
    for row in inv_ex:
        print(f"  {[round(v, 4) for v in row]}")

    # test matrice mare
    print("\n" + "=" * 60)
    print(f"6. TEST MATRICE MARE (n=100)")
    print("=" * 60)
    import time
    n_big = 100
    A_big = generate_matrix(n_big, seed=13)
    s_big = generate_vector_s(n_big, seed=17)
    b_big = compute_b(A_big, s_big)

    t0 = time.perf_counter()
    R_big, Qt_big, Qtb_big = householder_qr(A_big, b_big, eps)
    x_big = back_substitution(R_big, Qtb_big, eps)
    t1 = time.perf_counter()

    Ax_big = mat_vec(A_big, x_big)
    res_big = norm2(vec_diff(Ax_big, b_big)) #4
    rel_big = norm2(vec_diff(x_big, s_big)) / norm2(s_big) #4
    print(f"n=100: timp={t1-t0:.3f}s")
    print(f"  ||Ax - b||_2 = {res_big:.2e}")
    print(f"  ||x - s||/||s|| = {rel_big:.2e}")
    print(f"  OK: {'✓' if res_big < 1e-6 and rel_big < 1e-6 else '✗'}")


if __name__ == "__main__":
    main()
