import math
import os

DATA_DIR = "tema4_files"
K_MAX = 10000
DIV_MAX = 1e10

def read_vector(filepath):
    v = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                v.append(float(line))
    return v

# Cerinta 2

def find_p_q(n, len_d1, len_d2):
    return n - len_d1, n - len_d2

# Cerinta 4

def gauss_seidel(d0, d1, d2, b, p, q, eps):
    n = len(d0)

    # Cerinta 3:
    for i in range(n):
        if abs(d0[i]) <= eps:
            return None, f"Element zero pe diagonală la i={i}"

    xc = [0.0] * n   # x curent  x^(k+1)
    xp = [0.0] * n   # x anterior x^(k)

    k = 0
    delta_x = eps + 1.0  # intrarea în bucla

    while delta_x >= eps and k < K_MAX:
        for i in range(n):
            xp[i] = xc[i]

        # nou xc
        for i in range(n):
            s = b[i]

            # i-q < i => xc
            if i - q >= 0:
                s -= d2[i - q] * xc[i - q]

            # i-p < i => xc
            if i - p >= 0:
                s -= d1[i - p] * xc[i - p]

            # i+p > i => xp
            if i + p <= n - 1:
                s -= d1[i] * xp[i + p]

            # i+q > i => xp 
            if i + q <= n - 1:
                s -= d2[i] * xp[i + q]

            xc[i] = s / d0[i]

        delta_x = math.sqrt(sum((xc[i] - xp[i]) ** 2 for i in range(n)))
        k += 1

        if delta_x > DIV_MAX:
            return None, f"Divergență după {k} iterații, delta={delta_x:.2e}"

    if delta_x < eps:
        return xc, f"Convergență după {k} iterații, delta={delta_x:.2e}"
    else:
        return None, f"Limită iterații atinsă ({K_MAX}), delta={delta_x:.2e}"

# Cerinta 5

def mat_vec_sparse(d0, d1, d2, p, q, x):

    n = len(d0)
    y = [0.0] * n

    for i in range(n):
        # Diag princ
        y[i] += d0[i] * x[i]

        # Diag sec p
        if i < len(d1):            
            y[i]     += d1[i] * x[i + p]
            y[i + p] += d1[i] * x[i]

        # Diag sec q
        if i < len(d2):             
            y[i]     += d2[i] * x[i + q]
            y[i + q] += d2[i] * x[i]

    return y

def norm_inf(v):
    return max(abs(x) for x in v)

def norm2(v):
    return math.sqrt(sum(x * x for x in v))

def vec_diff(a, b):
    return [a[i] - b[i] for i in range(len(a))]

def exact_solution(idx, n):
    if idx == 1: return [2.0 / 3.0] * n
    if idx == 2: return [i * 4.0 / 7.0 for i in range(n)]
    if idx == 3: return [2.0] * n
    if idx == 4: return [float(i) for i in range(n)]
    if idx == 5: return [i * 0.4 for i in range(n)]

def process_system(idx, eps):
    print(f"\n{'=' * 60}")
    print(f"SISTEMUL {idx}")
    print(f"{'=' * 60}")

    d0 = read_vector(os.path.join(DATA_DIR, f"d0_{idx}.txt"))
    d1 = read_vector(os.path.join(DATA_DIR, f"d1_{idx}.txt"))
    d2 = read_vector(os.path.join(DATA_DIR, f"d2_{idx}.txt"))
    b  = read_vector(os.path.join(DATA_DIR, f"b_{idx}.txt"))

    # Cerinta 1
    n = len(d0)
    print(f"\n1. Dimensiunea sistemului: n = {n}")

    # Cerinta 2
    p, q = find_p_q(n, len(d1), len(d2))
    print(f"\n2. Diagonale secundare: p = {p},  q = {q}")
    print(f"   d1 are {len(d1)} elemente = n - p = {n - p}  {'✓' if len(d1) == n-p else '✗'}")
    print(f"   d2 are {len(d2)} elemente = n - q = {n - q}  {'✓' if len(d2) == n-q else '✗'}")
    print(f"\n   Relații teoretice:")
    print(f"   Diag. sup. ord p: a[0][p], a[1][p+1], ..., a[n-1-p][n-1]  => x = n-1-p = {n-1-p}")
    print(f"   Are n-p = {n-p} elemente  (verificat: len(d1) = {len(d1)})")
    print(f"   Diag. sup. ord q: a[0][q], a[1][q+1], ..., a[n-1-q][n-1]  => y = n-1-q = {n-1-q}")
    print(f"   Are n-q = {n-q} elemente  (verificat: len(d2) = {len(d2)})")
    # Cerinta 3
    zeros = [i for i in range(n) if abs(d0[i]) <= eps]
    if zeros:
        print(f"\n3. ✗ Elemente nule pe diagonală: {zeros[:5]}")
    else:
        print(f"\n3. ✓ Toate elementele din d0 sunt nenule  (min = {min(abs(x) for x in d0):.4f})")

    # Cerinta 4
    print(f"\n4. Gauss-Seidel (eps=1e-{int(-math.log10(eps))}, kmax={K_MAX})...")
    import time
    t0 = time.perf_counter()
    xGS, msg = gauss_seidel(d0, d1, d2, b, p, q, eps)
    t1 = time.perf_counter()
    print(f"   {msg}  [{t1-t0:.2f}s]")

    # Cerinta 5 + 6
    if xGS is None:
        print(f"\n5-6. Nu se calculează ||AxGS - b||_inf — soluția nu a putut fi aproximată (divergență).")
        return
    y = mat_vec_sparse(d0, d1, d2, p, q, xGS)
    residual = vec_diff(y, b)
    err_inf = norm_inf(residual)
    print(f"\n5-6. ||A*xGS - b||_inf = {err_inf:.2e}")
    
    # Comp cu sol exacta
    x_exact = exact_solution(idx, n)
    diff = vec_diff(xGS, x_exact)
    norm_s = norm2(x_exact) or 1.0
    rel_err = norm2(diff) / norm_s
    print(f"     ||xGS - x_exact||_2 / ||x_exact||_2 = {rel_err:.2e}")

    show = min(5, n)
    print(f"\n   Primele {show} valori:")
    print(f"   xGS   : {[round(xGS[i], 6) for i in range(show)]}")
    print(f"   exacte: {[round(x_exact[i], 6) for i in range(show)]}")


def main():
    print("=" * 60)
    print("TEMA 4 - Gauss-Seidel pentru matrice rare")
    print("=" * 60)

    try:
        t = int(input("\nIntroduceți t (epsilon = 10^(-t), t=5..9): "))
    except ValueError:
        print("Input invalid. Se folosește t=8.")
        t = 8

    eps = 10.0 ** (-t)
    print(f"epsilon = 10^(-{t}) = {eps}")

    for idx in range(1, 6):
        process_system(idx, eps)

    print(f"\n{'=' * 60}")
    print("GATA")


if __name__ == "__main__":
    main()
