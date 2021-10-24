from graphics import *

filename = "test_equ.txt"
def read_equation(filename):
    equations = []
    with open(f'{filename}', 'r') as f:

        for line in f.read().splitlines():
            equations.append(line)

    return equations

def main():
    win = GraphWin("My window", 500, 500)

    win.setBackground(color_rgb(255, 255, 255))

    equations = read_equation(filename=filename)

    for i in range(1, len(equations)):
        pt = Point()

    win.getMouse()
    win.close()

# main()
print(read_equation(filename))