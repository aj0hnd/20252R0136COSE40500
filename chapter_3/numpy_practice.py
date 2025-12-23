import numpy as np

# 다차원 배열
a = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8]
], dtype=np.float32)

print(f"dim: {a.ndim}")
print(f"shape: {a.shape}")

# 행렬곱
a = np.array([
    [1, 2],
    [5, 6],
], dtype=np.float32)
b = np.array([
    [1, 2],
    [5, 6],
], dtype=np.float32)

print(f"dot product:\n {np.dot(a, b)}")
print(f"dot product:\n {a.dot(b)}")