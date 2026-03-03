import math
import time
import random
import warnings

print("\nex1")
# ex1
def find_machine_epsilon():
    m = 0
    while True:
        u = 10.0 ** (-m)
        if 1.0 + u == 1.0:
            u_prev = 10.0 ** (-(m - 1))
            return u_prev, m - 1
        m += 1


u, m = find_machine_epsilon()
print(f"u = 10^(-{m}) = {u}")

print("\nex2")
# ex2
x, y, z = 1.0, u / 10, u / 10
lhs_add, rhs_add = (x + y) + z, x + (y + z)
print(f"Adunare neasociativă [(x + y) + z == x + (y + z)]: {lhs_add == rhs_add}")

a2, b2, c2 = 0.1, 0.2, 0.3
lhs_mul, rhs_mul = (a2 * b2) * c2, a2 * (b2 * c2)
print(f"Înmulțire neasociativă [(a * b) * c == a * (b * c)]: {lhs_mul == rhs_mul}")

print("\nex3")
# ex3
TAN_COEF_C1 = 1.0/3.0            
TAN_COEF_C2 = 2.0/15.0          
TAN_COEF_C3 = 17.0/315.0        
TAN_COEF_C4 = 62.0/2835.0       

def my_tan_continued_fraction(x, eps=1e-10):
    x_reduced = _reduce_to_half_pi(x)
    if x_reduced is None: return float('inf')
    x = x_reduced
    
    sign = -1.0 if x < 0 else 1.0
    x = abs(x)
    
    mic = 1e-12
    x2 = x * x
    f = C = mic
    D = 0.0
    j = 1
    
    while j < 10000:
        aj, bj = (x, 1.0) if j == 1 else (-x2, float(2 * j - 1))
        
        D = bj + aj * D
        if D == 0.0: D = mic
        
        C = bj + aj / C
        if C == 0.0: C = mic
        
        D = 1.0 / D
        delta = C * D
        f *= delta
        
        if abs(delta - 1.0) < eps: break
        j += 1
        
    return sign * f

def my_tan_polynomial(x):
    x_reduced = _reduce_to_half_pi(x)
    if x_reduced is None: return float('inf')
    x = x_reduced
    
    sign = -1.0 if x < 0 else 1.0
    x = abs(x)
    
    use_reciprocal = False
    if x > math.pi / 4:
        x = math.pi / 2 - x
        use_reciprocal = True
    
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x6 = x4 * x2
    
    P = TAN_COEF_C1 + TAN_COEF_C2 * x2 + TAN_COEF_C3 * x4 + TAN_COEF_C4 * x6
    result = x + x3 * P
    
    if use_reciprocal:
        if result == 0.0: return sign * float('inf')
        result = 1.0 / result
        
    return sign * result

def _reduce_to_half_pi(x):
    half_pi = math.pi / 2
    x = x % math.pi  
    
    if x > half_pi: x -= math.pi
    elif x < -half_pi: x += math.pi
    
    if abs(abs(x) - half_pi) < 1e-12: return None
    return x

print("Rulare benchmark pe 10.000 de valori...")

N = 10000
random.seed(42)
half_pi = math.pi / 2
xs = [random.uniform(-half_pi + 1e-6, half_pi - 1e-6) for _ in range(N)]

start_cf = time.perf_counter()
errors_cf = [abs(math.tan(x) - my_tan_continued_fraction(x)) for x in xs]
time_cf = time.perf_counter() - start_cf

start_poly = time.perf_counter()
errors_poly = [abs(math.tan(x) - my_tan_polynomial(x)) for x in xs]
time_poly = time.perf_counter() - start_poly

faster = "Polinom" if time_poly < time_cf else "Fracții continue"
more_accurate = "Fracții continue" if max(errors_cf) < max(errors_poly) else "Polinom"

print(f"Fracții continue : Timp = {time_cf:.4f}s | Eroare medie = {sum(errors_cf)/N:.2e} | Eroare maximă = {max(errors_cf):.2e}")
print(f"Polinom MacLaurin: Timp = {time_poly:.4f}s | Eroare medie = {sum(errors_poly)/N:.2e} | Eroare maximă = {max(errors_poly):.2e}")
