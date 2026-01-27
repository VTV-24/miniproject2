import os
import papermill as pm

# ========================
# PROJECT ROOT
# ========================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

def p(rel_path):
    return os.path.join(PROJECT_ROOT, rel_path)

print("PROJECT_ROOT =", PROJECT_ROOT)

# ========================
# OUTPUT DIR
# ========================
os.makedirs(p("notebooks/runs"), exist_ok=True)

KERNEL = "python3"

# ========================
# 1) Preprocessing + EDA
# ========================
pm.execute_notebook(
    p("notebooks/preprocessing_and_eda.ipynb"),
    p("notebooks/runs/preprocessing_and_eda_run.ipynb"),
    kernel_name=KERNEL,
)

# ========================
# 2) Feature + Semi dataset preparation
# (tạo luôn X_labeled, X_unlabeled, X_val, X_test)
# ========================
pm.execute_notebook(
    p("notebooks/feature_preparation.ipynb"),
    p("notebooks/runs/feature_preparation_run.ipynb"),
    kernel_name=KERNEL,
)

# ========================
# 3) Supervised baseline
# ========================
pm.execute_notebook(
    p("notebooks/classification_modelling.ipynb"),
    p("notebooks/runs/classification_modelling_run.ipynb"),
    kernel_name=KERNEL,
)

# ========================
# 4) Semi-supervised: Self-training
# ========================
pm.execute_notebook(
    p("notebooks/semi_self_training.ipynb"),
    p("notebooks/runs/semi_self_training_run.ipynb"),
    kernel_name=KERNEL,
)

# ========================
# 5) Semi-supervised: Co-training
# ========================
pm.execute_notebook(
    p("notebooks/semi_co_training.ipynb"),
    p("notebooks/runs/semi_co_training_run.ipynb"),
    kernel_name=KERNEL,
)

print("Đã chạy xong toàn bộ pipeline: baseline + self-training + co-training.")
