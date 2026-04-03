import math
import random

def copy_matrix(M):
    return [row[:] for row in M]

def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

def mat_mul(A, B):
    """Înmulțire matrice A (m x k) * B (k x n) => C (m x n)"""
    m, k, n = len(A), len(B), len(B[0])
    C = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0.0
            for l in range(k):
                s += A[i][l] * B[l][j]
            C[i][j] = s
    return C

def mat_sub(A, B):
    n = len(A)
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]

def mat_norm_frobenius(A):
    """||A||_F = sqrt(sum a_ij^2)"""
    return math.sqrt(sum(A[i][j]**2 for i in range(len(A)) for j in range(len(A[0]))))

def mat_norm_1(A):
    """||A||_1 = max coloane (suma pe coloană)"""
    m, n = len(A), len(A[0])
    return max(sum(abs(A[i][j]) for i in range(m)) for j in range(n))

def diag_matrix(vals):
    """Construiește o matrice diagonală din lista de valori"""
    n = len(vals)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        D[i][i] = vals[i]
    return D

def transpose(A):
    m, n = len(A), len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]

def generate_symmetric_matrix(n, seed=42):
    """Generează o matrice simetrică pozitiv definită A = B*B^T + n*I"""
    random.seed(seed)
    B = [[random.uniform(-2, 2) for _ in range(n)] for _ in range(n)]
    # A = B*B^T
    A = [[sum(B[i][k] * B[j][k] for k in range(n)) for j in range(n)] for i in range(n)]
    # Adăugăm n*I pentru pozitiv definiteness
    for i in range(n):
        A[i][i] += n
    return A

def print_matrix(A, name="A", max_size=6):
    n, m = len(A), len(A[0])
    d = min(n, max_size)
    suffix = f" [primele {d}x{d}]" if n > max_size else ""
    print(f"\nMatricea {name} ({n}x{m}){suffix}:")
    for i in range(d):
        row = "  [" + "".join(f"{A[i][j]:10.4f}" for j in range(min(m, d)))
        if m > max_size:
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

def find_max_offdiag(A, n):
    """
    Găsește indicii (p, q) ai celui mai mare element nediagonal
    în valoare absolută, căutând doar în triunghiul strict inferior
    (datorită simetriei).
    PDF formula (1): |a_pq| = max{|a_ij|; i>j}
    """
    p, q = 1, 0
    max_val = abs(A[1][0])
    for i in range(2, n):
        for j in range(i):
            if abs(A[i][j]) > max_val:
                max_val = abs(A[i][j])
                p, q = i, j
    return p, q, max_val

def compute_cs(A, p, q):
    """
    Calculează c = cos(θ) și s = sin(θ) pentru rotația Jacobi.
    PDF formula (3): α = (a_pp - a_qq) / (2 * a_pq)
    t = -α + semn(α) * sqrt(α² + 1)
    PDF formula (4): c = 1/sqrt(1+t²), s = t/sqrt(1+t²)
    """
    if abs(A[p][q]) < 1e-15:
        return 1.0, 0.0, 0.0  # c=1, s=0, t=0

    alpha = (A[p][p] - A[q][q]) / (2.0 * A[p][q])

    # Alegem rădăcina de modul minim
    if alpha >= 0:
        t = -alpha + math.sqrt(alpha**2 + 1)
    else:
        t = -alpha - math.sqrt(alpha**2 + 1)

    denom = math.sqrt(1.0 + t**2)
    c = 1.0 / denom
    s = t / denom
    return c, s, t

def jacobi_step(A, U, p, q, c, s, t, n):
    """
    Aplică un pas Jacobi: A = R_pq * A * R_pq^T și U = U * R_pq^T
    Folosește formulele (5) și (7) din PDF — fără matrice auxiliară.
    """
    for j in range(n):
        if j == p or j == q:
            continue
        apj = A[p][j]
        aqj = A[q][j]
        A[p][j] = c * apj + s * aqj   # b_pj = c*a_pj + s*a_qj
        A[q][j] = -s * apj + c * aqj  # b_qj = -s*a_pj + c*a_qj

        A[j][p] = A[p][j]
        A[j][q] = A[q][j]

    A[p][p] = A[p][p] + t * A[p][q]   # b_pp = a_pp + t*a_pq (înainte de a seta 0)
    A[q][q] = A[q][q] - t * A[p][q]   # b_qq = a_qq - t*a_pq
    A[p][q] = 0.0
    A[q][p] = 0.0

    for i in range(n):
        uip = U[i][p]
        uiq = U[i][q]
        U[i][p] = c * uip + s * uiq
        U[i][q] = -s * uip + c * uiq

def jacobi_eigenvalues(A_in, eps, kmax=10000):
    n = len(A_in)
    A = copy_matrix(A_in)
    U = identity(n)  

    k = 0
    while k < kmax:
        p, q, max_val = find_max_offdiag(A, n)

        if max_val <= eps:
            break

        c, s, t = compute_cs(A, p, q)

        jacobi_step(A, U, p, q, c, s, t, n)

        k += 1

    eigenvalues = [A[i][i] for i in range(n)]

    return eigenvalues, U, k

def cholesky_ldlt_lower(A_in, eps):
    n = len(A_in)
    A = copy_matrix(A_in)

    d = [0.0] * n
    for p in range(n):
        dp = A[p][p]
        for k in range(p):
            dp -= d[k] * A[p][k] * A[p][k]
        if abs(dp) <= eps:
            raise ValueError(f"Cholesky eșuat la p={p}: d_p={dp:.2e} (matrice nepozitiv definită?)")
        if dp < 0:
            raise ValueError(f"Cholesky eșuat: d_p={dp:.2e} < 0 (matrice nepozitiv definită)")
        d[p] = dp
        for i in range(p + 1, n):
            s = A[p][i]  
            for k in range(p):
                s -= d[k] * A[i][k] * A[p][k]
            A[i][p] = s / dp

    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i):
            L[i][j] = A[i][j] * math.sqrt(d[j])  # l_ij * sqrt(d_j)
        L[i][i] = math.sqrt(d[i])  # diagonala

    return L

def cholesky_iteration(A_in, eps, kmax=1000):
    n = len(A_in)
    A = copy_matrix(A_in)
    A_prev = copy_matrix(A)

    k = 0
    while k < kmax:
        try:
            L = cholesky_ldlt_lower(A, eps)
        except ValueError as e:
            print(f"   [Cholesky iteration] Oprire la k={k}: {e}")
            break

        Lt = transpose(L)
        A_prev = copy_matrix(A)
        A = mat_mul(Lt, L)
        
        diff = mat_norm_frobenius(mat_sub(A, A_prev))
        k += 1

        if diff < eps:
            break

    return A, k

def svd_analysis(A_in, eps):
    import numpy as np

    A = np.array(A_in, dtype=float)
    p, n = A.shape

    U, sigma, Vt = np.linalg.svd(A, full_matrices=True)
    V = Vt.T

    print(f"\n   Valori singulare σ_i:")
    for i, s in enumerate(sigma):
        print(f"   σ_{i+1} = {s:.6f}")

    rank = int(np.sum(sigma > eps))
    print(f"\n   Rang(A) = {rank}  (valori singulare > eps={eps})")

    sigma_pos = sigma[sigma > eps]
    cond = float(sigma_pos[0]) / float(sigma_pos[-1]) if len(sigma_pos) > 0 else float('inf')
    cond_lib = np.linalg.cond(A)
    print(f"\n   Număr condiționare k2(A) = σ_max/σ_min = {cond:.6f}")
    print(f"   Număr condiționare (numpy) = {cond_lib:.6f}")

    SI = np.zeros((n, p))
    for i in range(min(p, n)):
        if i < len(sigma) and sigma[i] > eps:
            SI[i][i] = 1.0 / sigma[i]

    A_pseudo = V @ SI @ U.T  # A^I = V * S^I * U^T
    print(f"\n   Pseudoinversa Moore-Penrose A^I ({n}x{p}) calculată.")

    AtA = A.T @ A
    try:
        AtA_inv = np.linalg.inv(AtA)
        A_lstsq = AtA_inv @ A.T
        diff_norm = np.linalg.norm(A_pseudo - A_lstsq, ord=1)
        print(f"   Pseudoinversa least-squares A^J ({n}x{p}) calculată.")
        print(f"\n   ||A^I - A^J||_1 = {diff_norm:.2e}")
    except np.linalg.LinAlgError:
        print("   A^T*A este singulară, A^J nu se poate calcula.")
        A_lstsq = None
        diff_norm = None

    return sigma.tolist(), rank, cond, A_pseudo.tolist()

def verify_jacobi(A_init, eigenvalues, U):
    n = len(A_init)
    Lambda = diag_matrix(eigenvalues)

    AU = mat_mul(A_init, U)       # A_init * U
    UL = mat_mul(U, Lambda)       # U * Λ

    diff = mat_sub(AU, UL)
    return mat_norm_frobenius(diff)

def main():
    print("=" * 60)
    print("TEMA 5 - Jacobi, Cholesky iteration, SVD")
    print("=" * 60)

    try:
        n = int(input("\nIntroduceți n (dimensiunea matricei pătratice): "))
        t = int(input("Introduceți t (epsilon = 10^(-t)): "))
    except ValueError:
        print("Input invalid. Se folosesc n=4, t=8.")
        n, t = 4, 8

    eps = 10.0 ** (-t)
    print(f"\nn = {n}, epsilon = 10^(-{t}) = {eps}")

    # --------------------------------------------------------
    # PARTEA 1: Metoda Jacobi (p = n, A simetrică)
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("PARTEA 1: METODA JACOBI (valori și vectori proprii)")
    print("=" * 60)

    A_sym = generate_symmetric_matrix(n, seed=42)
    A_init_copy = copy_matrix(A_sym)

    print_matrix(A_sym, "A (simetrică, pozitiv definită)")

    eigenvalues, U, iterations = jacobi_eigenvalues(A_sym, eps)

    print(f"\nJacobi a converge după {iterations} iterații.")
    print_vector(sorted(eigenvalues, reverse=True), "Valori proprii λ (sortate)")
    print_matrix(U, "U (vectori proprii pe coloane)")

    # Verificare: ||A_init * U - U * Λ||
    err_jacobi = verify_jacobi(A_init_copy, eigenvalues, U)
    print(f"\n||A_init * U - U * Λ||_F = {err_jacobi:.2e}  (target < 1e-6)")

    # Verificare cu numpy
    try:
        import numpy as np
        eigvals_np, eigvecs_np = np.linalg.eigh(np.array(A_init_copy))
        print(f"Valori proprii numpy (eigh): {[round(v,6) for v in sorted(eigvals_np.tolist(), reverse=True)]}")
        # Comparare valori proprii
        ev_jacobi_sorted = sorted(eigenvalues)
        ev_np_sorted = sorted(eigvals_np.tolist())
        err_ev = math.sqrt(sum((ev_jacobi_sorted[i]-ev_np_sorted[i])**2 for i in range(n)))
        print(f"||λ_Jacobi - λ_numpy||_2 = {err_ev:.2e}")
    except ImportError:
        pass

    print("\n--- Verificare exemple ---")
    examples = [
        ([[0,0,1],[0,0,1],[1,1,1]], "Ex1: valori proprii așteptate [-1, 0, 2]"),
        ([[1,1,2],[1,1,2],[2,2,2]], "Ex2: valori proprii așteptate [0, ~-0.828, ~4.828]"),
    ]
    for A_ex, desc in examples:
        ev_ex, U_ex, it_ex = jacobi_eigenvalues(A_ex, eps)
        print(f"\n{desc}")
        print(f"  Jacobi: {[round(v,4) for v in sorted(ev_ex, reverse=True)]} ({it_ex} iterații)")

    # --------------------------------------------------------
    # PARTEA 2: Șirul de matrice A(k) prin Cholesky
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("PARTEA 2: ȘIRUL A(k) PRIN CHOLESKY REPETAT")
    print("=" * 60)

    print("\nA(0) = A (matricea simetrică pozitiv definită de mai sus)")
    print("A(k+1) = L_k^T * L_k  unde  A(k) = L_k * L_k^T")

    A_chol_iter, k_chol = cholesky_iteration(A_init_copy, eps)

    print(f"\nConvergență după {k_chol} iterații.")
    print_matrix(A_chol_iter, "A_final (ultima matrice din șir)")

    diag_vals = [A_chol_iter[i][i] for i in range(n)]
    print_vector(sorted(diag_vals, reverse=True), "Diagonala A_final (valori proprii aproximative)")
    print(f"\nForma matricei finale: aproximativ DIAGONALĂ")
    print(f"Pe diagonală se găsesc valorile proprii ale matricei A_init.")

    # --------------------------------------------------------
    # PARTEA 3: SVD pentru p > n
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("PARTEA 3: SVD (p > n)")
    print("=" * 60)

    try:
        p_rows = int(input(f"\nIntroduceți p (numărul de linii, p > n={n}): "))
        if p_rows <= n:
            print(f"p trebuie să fie > n={n}. Se folosește p={n+3}.")
            p_rows = n + 3
    except ValueError:
        p_rows = n + 3
        print(f"Input invalid. Se folosește p={p_rows}.")

    # Generăm o matrice aleatoare p x n
    random.seed(7)
    A_rect = [[random.uniform(-3, 3) for _ in range(n)] for _ in range(p_rows)]
    print(f"\nMatrice A aleatoare ({p_rows}x{n}) generată.")
    print_matrix(A_rect, "A (p x n)")

    try:
        sigma, rank, cond, A_pseudo = svd_analysis(A_rect, eps)
        print(f"\nAnaliză SVD completă ✓")
    except ImportError:
        print("numpy indisponibil — instalați cu: pip install numpy")

    # --------------------------------------------------------
    # Test cu dimensiune mare
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST JACOBI n=5 (exemplu manual verificabil)")
    print("=" * 60)
    import time
    A_test = generate_symmetric_matrix(5, seed=99)
    A_test_copy = copy_matrix(A_test)
    t0 = time.perf_counter()
    ev_test, U_test, it_test = jacobi_eigenvalues(A_test_copy, eps)
    t1 = time.perf_counter()
    err_test = verify_jacobi(A_test, ev_test, U_test)
    print(f"n=5: {it_test} iterații, timp={t1-t0:.4f}s")
    print(f"||A_init*U - U*Λ||_F = {err_test:.2e}")
    print(f"OK: {'✓' if err_test < 1e-6 else '✗'}")


if __name__ == "__main__":
    main()