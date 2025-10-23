import math
import random
import sys

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
    from numpy import array, linalg

    # формируем матрицу 3x3 блока
    M = [[0.0] * 3 for _ in range(3)]
    F_block = [0.0] * 3

    # верхняя строка (k)
    for j in range(3):
        col = k + j
        M[0][j] = p[col] if k == k else (a[k] if col == k - 1 else b[k] if col == k else c[k] if col == k + 1 else 0)
    F_block[0] = f[k]

    # средняя строка (k+1)
    for j in range(3):
        col = k + j
        M[1][j] = a[k + 1] if col == k else b[k + 1] if col == k + 1 else c[k + 1] if col == k + 2 else 0
    F_block[1] = f[k + 1]

    # нижняя строка (k+2)
    for j in range(3):
        col = k + j
        M[2][j] = q[col] if k + 2 == l else (
            a[k + 2] if col == k + 1 else b[k + 2] if col == k + 2 else c[k + 2] if col == k + 3 else 0)
    F_block[2] = f[k + 2]

    try:
        x_block = linalg.solve(array(M), array(F_block))
    except linalg.LinAlgError:
        return [math.nan] * (n + 1)

    x[k] = x_block[0]
    x[k + 1] = x_block[1]
    x[k + 2] = x_block[2]

    # === ШАГ 6. Обратный ход слева ===
    for i in range(k - 1, 0, -1):
        x[i] = f[i] - c[i] * x[i + 1]

    # === ШАГ 7. Обратный ход справа ===
    if l < n:
        for i in range(l + 1, n + 1):
            x[i] = f[i] - a[i] * x[i - 1]

    return x


def compute_error(x, x_true, q=1e-12):
    delta = 0.0
    n = len(x) - 1
    for i in range(1, n + 1):
        if math.isnan(x[i]):
            return math.nan
        if abs(x_true[i]) > q:
            di = abs(x[i] - x_true[i]) / abs(x_true[i])
        else:
            di = abs(x[i] - x_true[i])
        delta = max(delta, di)
    return delta


def main():
    while True:
        print("Выберите режим:")
        print("1 - Ручной ввод")
        print("2 - Автотестирование")

        try:
            mode = int(input())
        except ValueError:
            print("Ошибка: Введите число.")
            continue

        if mode == 1:
            try:
                n = int(input("Введите размер матрицы n: "))
            except ValueError:
                print("Ошибка: n должно быть числом.")
                continue

            while True:
                try:
                    k = int(input("Введите номер k: "))
                except ValueError:
                    print("Ошибка: k должно быть числом.")
                    continue
                l = k + 2
                if 1 <= k <= n - 2:
                    break
                else:
                    print(f"Ошибка: k должно быть в диапазоне [1, {n - 2}]")

            try:
                with open("Вариант 3.txt", "r", encoding="utf-8") as fin:
                    data = list(map(float, fin.read().split()))
            except FileNotFoundError:
                print("Не удалось открыть файл")
                return
            except ValueError:
                print("Некорректные данные в файле")
                return

            a = [0.0] * (n + 1)
            b = [0.0] * (n + 1)
            c = [0.0] * (n + 1)
            p = [0.0] * (n + 1)
            q = [0.0] * (n + 1)
            f = [0.0] * (n + 1)

            idx = 0
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    current_Aij = data[idx]
                    idx += 1
                    if j == i - 1:
                        a[i] = current_Aij
                    elif j == i:
                        b[i] = current_Aij
                    elif j == i + 1:
                        c[i] = current_Aij
                    if i == k:
                        p[j] = current_Aij
                    elif i == l:
                        q[j] = current_Aij

            for i in range(1, n + 1):
                f[i] = data[idx]
                idx += 1

            x = solve_system(n, k, a[:], b[:], c[:], f[:], p[:], q[:])
            print("\n=== Решение ===")
            for i in range(1, n + 1):
                print(f"x[{i}] = {x[i]:.8f}")

            # Оценка точности
            x_true_aux = [1.0] * (n + 1)
            f_aux = [0.0] * (n + 1)

            for i in range(1, n + 1):
                Ax_true_i = 0.0
                if i != k and i != l:
                    if i > 1:
                        Ax_true_i += a[i] * x_true_aux[i - 1]
                    Ax_true_i += b[i] * x_true_aux[i]
                    if i < n:
                        Ax_true_i += c[i] * x_true_aux[i + 1]
                elif i == k:
                    Ax_true_i = sum(p[j] * x_true_aux[j] for j in range(1, n + 1))
                elif i == l:
                    Ax_true_i = sum(q[j] * x_true_aux[j] for j in range(1, n + 1))
                f_aux[i] = Ax_true_i

            x_temp = solve_system(n, k, a[:], b[:], c[:], f_aux[:], p[:], q[:])
            if any(math.isnan(x_temp[i]) for i in range(1, n + 1)):
                print("\nОценка точности не вычислена: система неразрешима.")
            else:
                delta_estimate = max(abs(x_temp[i] - 1.0) for i in range(1, n + 1))
                print(f"\nОценка точности = {delta_estimate:.4e}")


        elif mode == 2:
            n = int(input("Введите размер системы n: "))
            while True:
                k = int(input("Введите номер k: "))
                l = k + 2
                if 1 <= k <= n - 2:
                    break
                else:
                    print(f"Ошибка: k должно быть в диапазоне [1, {n - 2}]")

            M = int(input("Введите диапазон коэффициентов M ([-M, M]): "))

            total_error = 0.0
            total_delta = 0.0
            successful_tests = 0
            attempts = 0
            MAX_ATTEMPTS = 50
            print("\nРезультаты тестирования")

            while successful_tests < 10 and attempts < MAX_ATTEMPTS:
                attempts += 1
                x_true = [0.0] + [random.randint(-M, M) for _ in range(n)]
                a = [0.0] * (n + 1)
                b = [0.0] * (n + 1)
                c = [0.0] * (n + 1)
                f = [0.0] * (n + 1)
                p = [0.0] * (n + 1)
                q = [0.0] * (n + 1)

                for i in range(1, n + 1):
                    a[i] = 0 if i == 1 else random.randint(-M, M)
                    b[i] = random.randint(-M, M)
                    c[i] = 0 if i == n else random.randint(-M, M)
                    if abs(b[i]) <= abs(a[i]) + abs(c[i]) + 1.0:
                        b[i] = (abs(a[i]) + abs(c[i]) + 1.0) * (1 if b[i] >= 0 else -1)
                    if abs(b[i]) < 1e-9:
                        b[i] = 1.0

                for j in range(1, n + 1):
                    p[j] = random.randint(-M, M)
                    q[j] = random.randint(-M, M)

                for i in range(1, n + 1):
                    if i != k and i != l:
                        f[i] = (a[i] * x_true[i - 1] if i > 1 else 0) + b[i] * x_true[i] + (c[i] * x_true[i + 1] if i < n else 0)
                    elif i == k:
                        f[i] = sum(p[j] * x_true[j] for j in range(1, n + 1))
                    elif i == l:
                        f[i] = sum(q[j] * x_true[j] for j in range(1, n + 1))

                x = solve_system(n, k, a[:], b[:], c[:], f[:], p[:], q[:])
                if math.isnan(x[1]):
                    continue

                err = compute_error(x, x_true)
                total_error += err
                print(f"Тест {successful_tests + 1:2d}: Погрешность = {err:.4e}")
                successful_tests += 1

            if successful_tests > 0:
                print(f"\nСредняя относительная погрешность = {total_error / successful_tests:.4e}")
            else:
                print("\nВсе тесты сингулярны.")

        else:
            print("Неверный выбор режима.")

        choice = input("\nПовторить (y/n)? ").strip().lower()
        if choice != 'y':
            break

    print("\nПрограмма завершена.")


if __name__ == "__main__":
    main()
