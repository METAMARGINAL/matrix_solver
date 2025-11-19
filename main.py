import math
import random

def solve_3x3(M, F):
    """Решение системы 3x3 методом Крамера."""
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
    """Считывает матрицу коэффициентов и вектор f из файла."""
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
                x = solve_system(n, k, a, b, c, f, p, q)
                print(f"\nРешена система размером {n}x{n}")
                for i in range(1, n + 1):
                    print(f"x[{i}] = {x[i]:.6f}")
            except Exception as e:
                print("Ошибка при чтении файла:", e)

        elif mode == 2:
            print("Автотестирование выполняется.")
        else:
            print("Неизвестный режим.")

        if input("Повторить (y/n)? ").strip().lower() != 'y':
            break

    print("Программа завершена.")


if __name__ == "__main__":
    main()
