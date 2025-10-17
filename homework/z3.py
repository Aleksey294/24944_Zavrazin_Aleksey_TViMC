import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Функция для бутстрапа и построения ДИ
# -----------------------------
def bootstrap_ci(data, M=5000, alpha=0.05):
    n = len(data)
    boot_mads = np.empty(M)
    boot_vars = np.empty(M)

    for i in range(M):
        resample = np.random.choice(data, size=n, replace=True)
        med = np.median(resample)
        boot_mads[i] = np.median(np.abs(resample - med))
        boot_vars[i] = resample.var(ddof=1)

    lower_idx = int((alpha/2) * M)
    upper_idx = int((1 - alpha/2) * M)
    mad_ci = (np.sort(boot_mads)[lower_idx], np.sort(boot_mads)[upper_idx])
    var_ci = (np.sort(boot_vars)[lower_idx], np.sort(boot_vars)[upper_idx])

    return {
        "mad": np.median(np.abs(data - np.median(data))),
        "var": data.var(ddof=1),
        "mad_ci": mad_ci,
        "var_ci": var_ci,
        "boot_mads": boot_mads,
        "boot_vars": boot_vars
    }

# -----------------------------
# Создаём разные выборки
# -----------------------------
np.random.seed(42)
n = 10000

datasets = {
    "Normal": np.random.normal(loc=0, scale=1, size=n),
    "Exponential": np.random.exponential(scale=1.0, size=n),
    "Uniform": np.random.uniform(low=-2, high=2, size=n),
    "With Outliers": np.concatenate([np.random.normal(0, 1, size=n-10), np.random.normal(20, 1, size=10)])
}

# -----------------------------
# Считаем бутстрап и визуализируем
# -----------------------------
fig, axes = plt.subplots(4, 2, figsize=(12, 16))
alpha = 0.05

for i, (name, data) in enumerate(datasets.items()):
    res = bootstrap_ci(data, M=3000, alpha=alpha)

    # --- MAD ---
    axes[i, 0].hist(res["boot_mads"], bins=40, alpha=0.7, edgecolor='black')
    axes[i, 0].axvline(res["mad_ci"][0], color='red', linestyle='--')
    axes[i, 0].axvline(res["mad_ci"][1], color='red', linestyle='--')
    axes[i, 0].axvline(res["mad"], color='green', linestyle='-')
    axes[i, 0].set_title(f"{name} — Bootstrap MAD")

    # --- VAR ---
    axes[i, 1].hist(res["boot_vars"], bins=40, alpha=0.7, edgecolor='black')
    axes[i, 1].axvline(res["var_ci"][0], color='red', linestyle='--')
    axes[i, 1].axvline(res["var_ci"][1], color='red', linestyle='--')
    axes[i, 1].axvline(res["var"], color='green', linestyle='-')
    axes[i, 1].set_title(f"{name} — Bootstrap Variance")

plt.tight_layout()
plt.show()
