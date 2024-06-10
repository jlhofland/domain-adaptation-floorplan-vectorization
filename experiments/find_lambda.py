
import math

epochs = 100

for par in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    for e in [0, epochs / 10, epochs * 2 / 10, epochs * 3 / 10, epochs * 4 / 10, epochs * 5 / 10, epochs * 6 / 10, epochs * 7 / 10, epochs * 8 / 10, epochs * 9 / 10, epochs]:
        print(f'par: {par}, e: {e}', 2 / (1 + math.exp(-par * (e / 400))) - 1)
    