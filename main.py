import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


pd.set_option('display.precision', 4)
np.set_printoptions(precision=4)


filename = 'Вариант10.csv'
df = pd.read_csv(filename, index_col=0, encoding='utf-8')

X_values = df.columns.values.astype(int)
Y_values = df.index.values.astype(int)

data = df.replace('', None).values.astype(float)
P = np.array(data, dtype=float)


sum_known = np.nansum(P)
missing_value = 1 - sum_known
P[9, 9] = missing_value

print("="*60)
print("Задание 2: Пропущенная вероятность")
print("="*60)
print(f"P(ξ=70, η=93) = {missing_value:.6f}\n")


P_X = np.sum(P, axis=0)
P_Y = np.sum(P, axis=1)

F_X = np.cumsum(P_X)
F_Y = np.cumsum(P_Y)

print("="*60)
print("Задание 4: Маргинальные вероятности")
print("="*60)
print("P(ξ) (для ξ):", dict(zip(X_values, P_X)))
print("P(η) (для η):", dict(zip(Y_values, P_Y)))
print()


fig, axes = plt.subplots(2, 2, figsize=(12, 8))


axes[0, 0].bar(X_values, P_X, width=5, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('ξ')
axes[0, 0].set_ylabel('P(ξ = ξ_i)')
axes[0, 0].set_title('Гистограмма маргинального распределения ξ')
axes[0, 0].grid(True, alpha=0.3)


axes[0, 1].bar(Y_values, P_Y, width=5, edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('η')
axes[0, 1].set_ylabel('P(η = η_j)')
axes[0, 1].set_title('Гистограмма маргинального распределения η')
axes[0, 1].grid(True, alpha=0.3)


axes[1, 0].step(X_values, F_X, where='post', linewidth=2, color='red')
axes[1, 0].scatter(X_values, F_X, color='red', zorder=5)
axes[1, 0].set_xlabel('ξ')
axes[1, 0].set_ylabel('F_ξ(x)')
axes[1, 0].set_title('Функция распределения ξ')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)


axes[1, 1].step(Y_values, F_Y, where='post', linewidth=2, color='blue')
axes[1, 1].scatter(Y_values, F_Y, color='blue', zorder=5)
axes[1, 1].set_xlabel('η')
axes[1, 1].set_ylabel('F_η(y)')
axes[1, 1].set_title('Функция распределения η')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('marginal_plots.png', dpi=150)
plt.show()


print("\n" + "="*60)
print("Задание 6: Проверка независимости по определению")
print("="*60)
print("Определение: p_ij = P(ξ=ξ_i) · P(η=η_j) для всех i, j\n")

independent = True
mismatches = []

for i in range(len(X_values)):
    for j in range(len(Y_values)):
        product = P_X[i] * P_Y[j]
        diff = abs(P[j, i] - product)
        if diff > 1e-8:
            independent = False
            mismatches.append({
                'i': i, 'j': j,
                'x': X_values[i], 'y': Y_values[j],
                'p_ij': P[j, i],
                'p_i * q_j': product,
                'diff': diff
            })

if independent:
    print("  Равенство ВЫПОЛНЯЕТСЯ для всех 100 пар.")
    print("   p_ij = P(ξ=ξ_i) · P(η=η_j) для всех i, j")
    print("\n  Следовательно, случайные величины ξ и η НЕЗАВИСИМЫ.")
else:
    print(f"Найдено {len(mismatches)} несовпадений:")
    for m in mismatches[:5]:
        print(f"   ξ={m['x']}, η={m['y']}: p_ij={m['p_ij']:.6f}, P(ξ)·P(η)={m['p_i * q_j']:.6f}, разница={m['diff']:.2e}")
    print("\n  Следовательно, ξ и η ЗАВИСИМЫ.")


E_X = np.sum(X_values * P_X)
E_Y = np.sum(Y_values * P_Y)

E_X2 = np.sum(X_values**2 * P_X)
E_Y2 = np.sum(Y_values**2 * P_Y)
D_X = E_X2 - E_X**2
D_Y = E_Y2 - E_Y**2
std_X = np.sqrt(D_X)
std_Y = np.sqrt(D_Y)


E_XY = 0
for i in range(len(X_values)):
    for j in range(len(Y_values)):
        E_XY += X_values[i] * Y_values[j] * P[j, i]

Cov_XY = E_XY - E_X * E_Y
corr = Cov_XY / (std_X * std_Y)

print("\n" + "="*60)
print("Задание 7: Коэффициент корреляции")
print("="*60)
print(f"r = {corr:.10f}")
if abs(corr) < 1e-8:
    print("  Коэффициент корреляции РАВЕН нулю.")
    if independent:
        print("   Причина: величины независимы (для независимых cov=0).")
    else:
        print("   Причина: отсутствие линейной зависимости (но есть нелинейная).")
else:
    print(f"  Коэффициент корреляции НЕ РАВЕН нулю (r = {corr:.6f})")

print("\n" + "="*60)
print("Задание 8: Уравнение регрессии E(η|ξ=ξ_i)")
print("="*60)

E_Y_given_X = np.zeros(len(X_values))
for i in range(len(X_values)):
    if P_X[i] > 0:
        E_Y_given_X[i] = np.sum(Y_values * P[:, i]) / P_X[i]

print("  ξ_i   |  E(η|ξ=ξ_i)")
print("  ------|-------------")
for x, ey in zip(X_values, E_Y_given_X):
    print(f"  {x:4d}  |  {ey:.6f}")

if independent:
    print(f"\n  Поскольку ξ и η независимы, E(η|ξ=ξ_i) = E(η) = {E_Y:.6f} для всех ξ_i.")

with open('results.txt', 'w', encoding='utf-8') as f:
    f.write("Результаты расчетов по варианту 10\n")
    f.write("="*60 + "\n")
    f.write(f"Пропущенная вероятность: {missing_value:.6f}\n\n")
    f.write(f"E(ξ) = {E_X:.6f}\n")
    f.write(f"E(η) = {E_Y:.6f}\n")
    f.write(f"D(ξ) = {D_X:.6f}\n")
    f.write(f"D(η) = {D_Y:.6f}\n")
    f.write(f"σ_ξ = {std_X:.6f}\n")
    f.write(f"σ_η = {std_Y:.6f}\n")
    f.write(f"Корреляция r = {corr:.10f}\n")
    f.write(f"Независимы: {independent}\n")

print("\n  Результаты сохранены в 'results.txt'")