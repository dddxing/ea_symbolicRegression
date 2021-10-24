from graphics import *

def main():
    win = GraphWin("My window", 500, 500)

    win.setBackground(color_rgb(255, 255, 255))

    win.getMouse()
    win.close()

main()