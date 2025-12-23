import numpy as np

class MulLayer:
    def __init__(self, name: str = ''):
        self.name = name
    
    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y
    
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy
    

class AddLayer:
    def __init__(self, name: str = ''):
        self.name = name

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x + y
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy

if __name__ == '__main__':
    num_apple, num_orange = 2, 3
    per_apple, per_orange = 100, 150
    tax_ratio = 1.1

    apple_integrater, orange_intergrater = MulLayer(), MulLayer()
    fruit_integrater, tax_calculator = AddLayer(), MulLayer()

    # forward
    apple_integrated = apple_integrater.forward(num_apple, per_apple)
    orange_integrated = orange_intergrater.forward(num_orange, per_orange)
    fruit_integrated = fruit_integrater.forward(apple_integrated, orange_integrated)
    final_out = tax_calculator.forward(fruit_integrated, tax_ratio)

    print(f"final output of 2 apples and 3 oranges: {final_out:.4f}\n")

    # backward
    dout = 1
    d_fruit, d_tax= tax_calculator.backward(dout)
    print(f"Gradient of fruit and tax: {d_fruit:.4f} / {d_tax:.4f}")
    
    d_apples, d_oranges = fruit_integrater.backward(dout=d_fruit)
    print(f"Gradient of integrated apples and oranges: {d_apples:.4f} / {d_oranges:.4f}")

    d_num_apple, d_per_apple = apple_integrater.backward(dout=d_apples)
    d_num_orange, d_per_orange = orange_intergrater.backward(dout=d_oranges)
    print(f"Gradient of num apple and per apple: {d_num_apple:.4f} / {d_per_apple:.4f}")
    print(f"Gradient of num orange and per orange: {d_num_orange:.4f} / {d_per_orange:.4f}")