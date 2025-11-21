import math
import random

# Решение системы 3x3 методом Крамера
def solve_3x3(M, F):
    def det3(m):
        return (
            m[0][0] * m[1][1] * m[2][2]
            + m[0][1] * m[1][2] * m[2][0]
            + m[0][2] * m[1][0] * m[2][1]
            - m[0][2] * m[1][1] * m[2][0]
            - m[0][0] * m[1][2] * m[2][1]
            - m[0][1] * m[1][0] * m[2][2]
        )

    D = det3(M)
    if abs(D) < 1e-18:
        return [math.nan, math.nan, math.nan]

    def replace_col(mat, col_index, vec):
        return [
            [vec[i] if j == col_index else mat[i][j] for j in range(3)]
            for i in range(3)
        ]

    Dx = det3(replace_col(M, 0, F))
    Dy = det3(replace_col(M, 1, F))
    Dz = det3(replace_col(M, 2, F))

    return [Dx / D, Dy / D, Dz / D]


def solve_system(n, k, a, b, c, f, p, q):
    l = k + 2
    x = [0.0] * (n + 1)

    # === ШАГ 1. Прямой ход слева ===
    for i in range(1, k):
        if abs(b[i]) < 1e-18:
            return [math.nan] * (n + 1)
        alpha = 1.0 / b[i]
        ci = c[i] * alpha
        fi = f[i] * alpha
        b[i] = 1.0
        c[i] = ci
        f[i] = fi
        if (i + 1) != k and (i + 1) != l and (i + 1) <= n:
            aold = a[i + 1]
            b[i + 1] -= aold * ci
            f[i + 1] -= aold * fi
            a[i + 1] = 0.0
        pold = p[i]
        if pold != 0.0:
            f[k] -= pold * fi
            p[i + 1] -= pold * ci
            p[i] = 0.0
        qold = q[i]
        if qold != 0.0:
            f[l] -= qold * fi
            q[i + 1] -= qold * ci
            q[i] = 0.0

    # === ШАГ 2. Прямой ход справа ===
    if l < n:
        for i in range(n, l, -1):
            if abs(b[i]) < 1e-18:
                return [math.nan] * (n + 1)
            beta = 1.0 / b[i]
            ai = a[i] * beta
            fi = f[i] * beta
            a[i] = ai
            b[i] = 1.0
            f[i] = fi
            if (i - 1) != k and (i - 1) != l and (i - 1) >= 1:
                cold = c[i - 1]
                b[i - 1] -= cold * ai
                f[i - 1] -= cold * fi
                c[i - 1] = 0.0
            pold = p[i]
            if pold != 0.0:
                f[k] -= pold * fi
                p[i - 1] -= pold * ai
                p[i] = 0.0
            qold = q[i]
            if qold != 0.0:
                f[l] -= qold * fi
                q[i - 1] -= qold * ai
                q[i] = 0.0

    # === ШАГ 3. Прямой ход внутри блока ===
    for i in range(k + 1, l):
        if abs(b[i]) < 1e-18:
            return [math.nan] * (n + 1)
        alpha = 1.0 / b[i]
        ci = c[i] * alpha
        fi = f[i] * alpha
        ai = a[i] * alpha
        a[i] = ai
        b[i] = 1.0
        c[i] = ci
        f[i] = fi
        if (i + 1) != k and (i + 1) != l and (i + 1) <= n:
            aold = a[i + 1]
            b[i + 1] -= aold * ci
            f[i + 1] -= aold * fi
            a[i + 1] = 0.0
        pold = p[i]
        if pold != 0.0:
            f[k] -= pold * fi
            p[i + 1] -= pold * ci
            p[i - 1] -= pold * ai
            p[i] = 0.0
        qold = q[i]
        if qold != 0.0:
            f[l] -= qold * fi
            q[i + 1] -= qold * ci
            q[i - 1] -= qold * ai
            q[i] = 0.0

    # === ШАГ 4. Решение блока 3x3 напрямую ===
    M = [[0.0] * 3 for _ in range(3)]
    F_block = [0.0] * 3

    for j in range(3):
        col = k + j
        M[0][j] = p[col]
    F_block[0] = f[k]

    for j in range(3):
        col = k + j
        if col == k:
            M[1][j] = a[k + 1]
        elif col == k + 1:
            M[1][j] = b[k + 1]
        elif col == k + 2:
            M[1][j] = c[k + 1]
        else:
            M[1][j] = 0.0
    F_block[1] = f[k + 1]

    for j in range(3):
        col = k + j
        M[2][j] = q[col]
    F_block[2] = f[k + 2]

    x_block = solve_3x3(M, F_block)
    if any(math.isnan(v) for v in x_block):
        return [math.nan] * (n + 1)

    x[k], x[k + 1], x[k + 2] = x_block

    # === ШАГ 5. Обратный ход слева ===
    for i in range(k - 1, 0, -1):
        x[i] = f[i] - c[i] * x[i + 1]

    # === ШАГ 6. Обратный ход справа ===
    if l < n:
        for i in range(l + 1, n + 1):
            x[i] = f[i] - a[i] * x[i - 1]

    return x


def read_system_from_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    empty_idx = None
    for i, line in enumerate(lines):
        if line == "":
            empty_idx = i
            break

    if empty_idx is None:
        matrix_lines = lines[:-10]
        f_lines = lines[-10:]
    else:
        matrix_lines = lines[:empty_idx]
        f_lines = lines[empty_idx + 1:]

    matrix = [list(map(float, line.split())) for line in matrix_lines]
    f_vec = [float(x) for x in f_lines]
    n = len(matrix)

    a = [0.0] * (n + 1)
    b = [0.0] * (n + 1)
    c = [0.0] * (n + 1)
    p = [0.0] * (n + 1)
    q = [0.0] * (n + 1)
    f = [0.0] + f_vec

    dens_rows = [i for i, row in enumerate(matrix, 1) if sum(1 for v in row if abs(v) > 1e-9) > 3]
    if len(dens_rows) < 2:
        raise ValueError("Не удалось определить блок 3x3 (k,l).")
    k, l = dens_rows[0], dens_rows[1]

    for i in range(1, n + 1):
        if i != k and i != l:
            if i > 1:
                a[i] = matrix[i - 1][i - 2]
            b[i] = matrix[i - 1][i - 1]
            if i < n:
                c[i] = matrix[i - 1][i]
        elif i == k:
            p[1:] = matrix[i - 1]
        elif i == l:
            q[1:] = matrix[i - 1]

    return n, k, a, b, c, f, p, q


def compute_rel_error(x_calc, x_true):
    n = len(x_calc) - 1
    q_ = 1e-12
    max_err = 0.0
    for i in range(1, n + 1):
        if abs(x_true[i]) > q_:
            err = abs(x_calc[i] - x_true[i]) / abs(x_true[i])
        else:
            err = abs(x_calc[i] - x_true[i])
        if err > max_err:
            max_err = err
    return max_err


# Исправленная оценка точности — максимальная относительная ошибка
def compute_precision_modified(x_calc, x_true):
    q_ = 1e-12
    delta = 0.0
    for xi, xi_hat in zip(x_true[1:], x_calc[1:]):
        if abs(xi) > q_:
            sol = abs(xi - xi_hat) / abs(xi)
        else:
            sol = abs(xi - xi_hat)
        if sol > delta:
            delta = sol
    return delta


def main():
    while True:
        print("Выберите режим:")
        print("1 - Считать систему из файла (input.txt)")
        print("2 - Автотестирование")

        try:
            mode = int(input())
        except ValueError:
            print("Ошибка: введите число.")
            continue

        if mode == 1:
            filename = "input.txt"
            try:
                n, k, a, b, c, f, p, q = read_system_from_file(filename)
                unit_vec = [0.0] + [1.0] * n

                l = k + 2
                f_tilde = [0.0] * (n + 1)
                for i in range(1, n + 1):
                    if i != k and i != l:
                        f_tilde[i] = (a[i] * unit_vec[i - 1] if i > 1 else 0) \
                                     + b[i] * unit_vec[i] \
                                     + (c[i] * unit_vec[i + 1] if i < n else 0)
                    elif i == k:
                        f_tilde[i] = sum(p[j] * unit_vec[j] for j in range(1, n + 1))
                    elif i == l:
                        f_tilde[i] = sum(q[j] * unit_vec[j] for j in range(1, n + 1))

                x = solve_system(n, k, a[:], b[:], c[:], f_tilde[:], p[:], q[:])

                print(f"\nРешена система с f = A * единичный вектор, размером {n}x{n}")
                for i in range(1, n + 1):
                    print(f"x[{i}] = {x[i]:.6f}")

                d_ = max(abs(1.0 - x[i]) for i in range(1, n + 1))
                print(f"\nОценка точности (max |1 - x_i|): {d_:.2e}")

                print("\nПервые 5 компонент (отклонения):")
                for i in range(1, min(6, n + 1)):
                    print(f"x[{i}] = {x[i]:.6f}, отклонение = {x[i] - 1.0:+.2e}")

            except Exception as e:
                print("Ошибка при чтении файла:", e)

        elif mode == 2:
            print("=== Автотестирование ===")
            test_cases = [
                {"n": 10, "M": 10},
                {"n": 10, "M": 100},
                {"n": 10, "M": 1000},
                {"n": 100, "M": 10},
                {"n": 100, "M": 100},
                {"n": 100, "M": 1000},
                {"n": 1000, "M": 10},
                {"n": 1000, "M": 100},
                {"n": 1000, "M": 1000},
            ]

            print(f"{'№':^3} {'n':^6} {'Диапазон M':^12} {'Отн. погрешность':^20} {'Оценка точности':^20}")
            print("-" * 65)

            for idx, case in enumerate(test_cases, 1):
                n = case["n"]
                M = case["M"]
                k = random.randint(1, n - 3)
                total_rel_err = 0.0
                total_prec = 0.0
                valid_tests = 0

                for _ in range(10):
                    x_true = [0.0] + [random.uniform(-M, M) for _ in range(n)]
                    a = [0.0] * (n + 1)
                    b = [0.0] * (n + 1)
                    c = [0.0] * (n + 1)
                    f = [0.0] * (n + 1)
                    p = [0.0] * (n + 1)
                    q = [0.0] * (n + 1)

                    for i in range(1, n + 1):
                        a[i] = 0 if i == 1 else random.uniform(-M, M)
                        b[i] = random.uniform(-M, M)
                        c[i] = 0 if i == n else random.uniform(-M, M)
                        if abs(b[i]) <= abs(a[i]) + abs(c[i]) + 1.0:
                            b[i] = (abs(a[i]) + abs(c[i]) + 1.0) * (1 if b[i] >= 0 else -1)
                        if abs(b[i]) < 1e-9:
                            b[i] = 1.0

                    for j in range(1, n + 1):
                        p[j] = random.uniform(-M, M)
                        q[j] = random.uniform(-M, M)

                    for i in range(1, n + 1):
                        if i != k and i != k + 2:
                            f[i] = (
                                (a[i] * x_true[i - 1] if i > 1 else 0)
                                + b[i] * x_true[i]
                                + (c[i] * x_true[i + 1] if i < n else 0)
                            )
                        elif i == k:
                            f[i] = sum(p[j] * x_true[j] for j in range(1, n + 1))
                        elif i == k + 2:
                            f[i] = sum(q[j] * x_true[j] for j in range(1, n + 1))

                    x_calc = solve_system(n, k, a[:], b[:], c[:], f[:], p[:], q[:])

                    if any(math.isnan(v) for v in x_calc):
                        continue

                    rel_err = compute_rel_error(x_calc, x_true)
                    delta = compute_precision_modified(x_calc, x_true)

                    total_rel_err += rel_err
                    total_prec += delta
                    valid_tests += 1

                if valid_tests > 0:
                    avg_rel_err = total_rel_err / valid_tests
                    avg_prec = total_prec / valid_tests
                    print(f"{idx:^3} {n:^6} {M:^12} {avg_rel_err:>20.3e} {avg_prec:>20.3e}")
                else:
                    print(f"{idx:^3} {n:^6} {M:^12} {'сингулярная система':^40}")

        else:
            print("Неизвестный режим.")

        if input("Повторить (y/n)? ").strip().lower() != 'y':
            break

    print("Программа завершена.")


if __name__ == "__main__":
    main()
