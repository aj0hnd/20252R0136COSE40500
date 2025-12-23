import numpy as np

# np.array(): 넘파이 배열(np.ndarray) 직접 생성
print('\n===== np.array() =====\n')
arr = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
], dtype=np.float32)
print(f"type: {type(arr)} --- shape: {arr.shape} --- data_type: {arr.dtype}\n")
# print(f"element:\n{arr}\n")

# basic calculation
print('\n===== basic calculation =====\n')
arr1 = np.array([[1, 2, 3]], dtype=np.float32)
arr2 = np.array([[4, 5, 6]], dtype=np.float32)
print(f"origin: {arr1} {arr2}")
print(f"add: {arr1 + arr2}")
print(f"extract: {arr1 - arr2}")
print(f"multiple: {arr1 * arr2}")
print(f"devide: {arr1 / arr2}")

# broadcast
print('\n===== broadcast =====\n')
cons, arr = 10, np.array([1, 4, 7], dtype=np.float32)
print(f"origin: {arr}")
print(f"+{cons}: {arr + cons}")
print(f"-{cons}: {arr - cons}")
print(f"*{cons}: {arr * cons}")
print(f"/{cons}: {arr / cons}")
print(f"%{cons}: {arr % cons}")

# boolean indexing
print('\n===== boolean indexing =====\n')
arr1, arr2 = np.linspace(start=0, stop=9, num=10), np.linspace(start=15, stop=6, num=10)
print(f"origin: {arr1} {arr2}")
print(f"bigger than 5: {arr1 > 5}")
print(f"index to arr2 : {arr2[arr1 > 5]}")