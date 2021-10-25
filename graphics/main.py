from graphics import *
import math
import random

filename = "ea_dm_equation.txt"
levels = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]

def read_equation(filename):
    equations = []
    with open(f'{filename}', 'r') as f:

        for line in f.read().splitlines():
            equations.append(line)

    return equations


def build_tree(binary_heap, win):

    ptr = 1
    radius = 20

    locations = [[0, 0], [2500, 50]]

    print(locations[ptr][0])
    print(locations[ptr][1])

    while ptr < len(binary_heap):
        x = locations[ptr][0]
        y = locations[ptr][1]
        # print(f"x = {x}, y = {y}")
        pt = Point(x, y)
        cir = Circle(pt, radius)
        txt = Text(pt, binary_heap[ptr])

        level = math.floor(math.log2(ptr))

        print(binary_heap[ptr], level)
        cir.draw(win)
        txt.draw(win)

        child_lt = [x - (1 * levels[level]), y + 100]
        child_rt = [x + (1 * levels[level]), y + 100]

        if ptr < len(binary_heap) // 2:
            line_lt = Line(Point(x,y+radius), Point(child_lt[0], child_lt[1]-radius))
            line_rt = Line(Point(x,y+radius), Point(child_rt[0], child_rt[1]-radius))

            line_lt.draw(win)
            line_rt.draw(win)

        locations.append(child_lt)
        locations.append(child_rt)

        ptr += 1
    # print(len(locations))

def main():
    win = GraphWin("My window", 5000, 2000)

    win.setBackground(color_rgb(255, 255, 255))
    bh = read_equation(filename=filename)
    build_tree(bh, win=win)

    win.getMouse()
    win.close()


main()