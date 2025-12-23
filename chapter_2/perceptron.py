class BasicPerceptron:
    def __init__(self, num_input: int = 2, name: str = 'basic'):
        self.num_input = num_input
        self.name = name

        self.and_params = {'w1': 0.5, 'w2': 0.5, 'theta': 0.9}
        self.nand_params = {'w1': -0.5, 'w2': -0.5, 'theta': -0.9}
        self.or_params = {'w1': 0.3, 'w2': 0.3, 'theta': 0.2}

    def get_y(self, x1, x2, params):
        return params['w1'] * x1 + params['w2'] * x2 - params['theta']

    def get_and(self, inputs: list):
        assert len(inputs) == 2, f"only two inputs are previded in this version."
        return 1 if self.get_y(*inputs, self.and_params) > 0 else 0
    
    def get_nand(self, inputs: list):
        assert len(inputs) == 2, f"only two inputs are previded in this version."
        return 1 if self.get_y(*inputs, self.nand_params) > 0 else 0

    def get_or(self, inputs: list):
        assert len(inputs) == 2, f"only two inputs are previded in this version."
        return 1 if self.get_y(*inputs, self.or_params) > 0 else 0

    def get_xor(self, inputs: list):
        h1 = self.get_nand(inputs)
        h2 = self.get_or(inputs)
        return self.get_and([h1, h2])

if __name__ == '__main__':
    perceptron = BasicPerceptron()

    print('=== AND Gate ===')
    for x1 in range(2):
        for x2 in range(2):
            print(f"AND of {x1}, {x2} : {perceptron.get_and([x1, x2])}")
    print()

    print('=== NAND Gate ===')
    for x1 in range(2):
        for x2 in range(2):
            print(f"NAND of {x1}, {x2} : {perceptron.get_nand([x1, x2])}")
    print()

    print('=== OR Gate ===')
    for x1 in range(2):
        for x2 in range(2):
            print(f"AND of {x1}, {x2} : {perceptron.get_or([x1, x2])}")
    print()

    print('=== XOR Gate ===')
    for x1 in range(2):
        for x2 in range(2):
            print(f"AND of {x1}, {x2} : {perceptron.get_xor([x1, x2])}")
    print()