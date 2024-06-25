# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 08:45:19 2024 under Python 3.11.7

@author: f24lerou
"""

# 8<--------------------------- Import modules ---------------------------

import numpy as np

from ifmta.tools import GetCartesianCoordinates

# 8<------------------------- Functions definitions ----------------------


def Cross(cross_size, *, center=[0, 0], width=1, **kargs):
    """
    cross : generate a cross over a square or rectangular support

    Author : Francois Leroux
    Contact : francois.leroux.pro@gmail.com
    Status : in progress
    Last update : 2024.02.28, Brest
    Comments : for even support size, coordinates are defined like [-2,-1,0,1] (N = 4)

    Inputs : MANDATORY : cross_size {integer}[pixel]

              OPTIONAL :  support_size : resolution of the support {tupple (1x2)}[pixel] - default value : cross_size
                          center : cartesian coordinates {tupple (1x2)}[pixel] - default value : [0,0]
                          width : width of the cross {integer}[pixel] - default value : 1

    Outputs : a binary cross
    """

    # read optinal parameters values
    support_size = kargs.get("support_size", [cross_size, cross_size])

    cross = np.zeros(support_size)

    cross[
        support_size[0] // 2
        - center[0]
        - width // 2 : support_size[0] // 2
        - center[0]
        + width // 2
        + width % 2,
        support_size[1] // 2
        + center[1]
        - cross_size // 2 : support_size[1] // 2
        + center[1]
        + cross_size // 2
        + cross_size % 2,
    ] = 1

    cross[
        support_size[0] // 2
        - center[0]
        - cross_size // 2 : support_size[0] // 2
        - center[0]
        + cross_size // 2
        + cross_size % 2,
        support_size[1] // 2
        + center[1]
        - width // 2 : support_size[1] // 2
        + center[1]
        + width // 2
        + width % 2,
    ] = 1

    return cross


def CrossDiag(n_points, *, width=0, **kargs):
    """
    CrossDiag : generate a diagonal cross over a square or rectangular support

    Author : Francois Leroux
    Contact : francois.leroux.pro@gmail.com
    Status : in progress
    Last update : 2024.02.28, Brest
    Comments : for even support size, coordinates are defined like [-2,-1,0,1] (N = 4)

    Inputs : MANDATORY : n_points : number of sample points in the X direction

              OPTIONAL :  width : width of the cross {integer}[pixel] - default value : 1

    Outputs : a binary diagonal cross
    """

    cross = np.zeros((n_points, n_points))
    for k in range(n_points):
        cross[k - width : k + width + 1, k - width : k + width + 1] = 1

    return cross


def GridSquares(nPointsX, *, spacing=1, width=1, margin=0, **kargs):
    """
    gridSquares : generate a grid of squares

    Author : Francois Leroux
    Contact : francois.leroux.pro@gmail.com
    Status : in progress
    Last update : 2024.03.06, Brest
    Comments : for even support size, coordinates are defined like [-2,-1,0,1] (N = 4)

    Inputs : MANDATORY : nPointsX {int} : number of squares in the X direction

              OPTIONAL :  nPointsY {int} : number of squares in the Y direction
                          spacing {int}[pixel] : space between two adjacent squares in the x or y direction - default value : 1
                          width : side length of the squares {integer}[pixel] - default value : 1
                          margin {int}[pixel] : space between the edges of the array and the firsts and lasts squares

    Outputs : a grid of squares
    """

    # read optinal parameters values
    nPointsY = kargs.get("nPointsY", nPointsX)

    point = np.ones([width, width])

    grid = np.zeros(
        [
            nPointsY * width + (nPointsY - 1) * spacing + 2 * margin,
            nPointsX * width + (nPointsX - 1) * spacing + 2 * margin,
        ]
    )

    for i in range(nPointsY):
        for j in range(nPointsX):
            grid[
                margin + i * (width + spacing) : margin + i * (width + spacing) + width,
                margin + j * (width + spacing) : margin + j * (width + spacing) + width,
            ] = point

    return grid


def Stick(n_points, *, width=1):
    """
    arrow : generate a stick

    Author : Francois Leroux
    Contact : francois.leroux.pro@gmail.com
    Status : in progress
    Last update : 2024.04.16, Brest
    Comments :

    Inputs : MANDATORY : nPointsX {int} : number of sampling points in the X direction

    Outputs : a stick
    """

    stick = np.zeros((n_points, n_points))
    stick[n_points // 2 - width // 2 : n_points // 2 + width // 2 + width % 2, :] = 1

    return stick


def DoubleArrow(n_points, *, arrow_heigth=1, arrow_width=0, fill=0):
    """
    arrow : generate an arrow

    Author : Francois Leroux
    Contact : francois.leroux.pro@gmail.com
    Status : in progress
    Last update : 2024.04.05, Brest
    Comments : bug for plt.imshow(arrow(5, arrow_heigth=2))

    Inputs : MANDATORY : nPointsX {int} : number of sampling points in the X direction

    Outputs : an arrow
    """

    arrow = np.zeros((n_points, n_points))

    arrow[n_points // 2, :] = 1
    arrow[
        n_points // 2 - arrow_width : n_points // 2 + arrow_width + 1,
        arrow_width : n_points - arrow_width,
    ] = 1

    for k in range(n_points // 2 - arrow_heigth, n_points // 2 + arrow_heigth + 1):
        arrow[k, k - n_points // 2 - 1 + k // (n_points // 2)] = 1
        arrow[k, -k + n_points // 2 - 1 + (n_points - k - 1) // (n_points // 2)] = 1

    if fill:
        arrow[
            n_points // 2 - arrow_heigth : n_points // 2 + arrow_heigth + 1,
            arrow_heigth,
        ] = 1
        arrow[
            n_points // 2 - arrow_heigth : n_points // 2 + arrow_heigth + 1,
            n_points - arrow_heigth - 1,
        ] = 1

    return arrow


def Disk(n_points):
    """
    disk : generate a disk

    Author : Francois Leroux
    Contact : francois.leroux.pro@gmail.com
    Status : in progress
    Last update : 2024.04.05, Brest
    Comments :

    Inputs : MANDATORY : nPointsX {int} : number of sampling points in the X direction

    Outputs : a disk
    """

    disk = np.zeros((n_points, n_points))

    [X, Y] = GetCartesianCoordinates(n_points)

    radial_coordinate = (X**2 + Y**2) ** 0.5
    disk[radial_coordinate < n_points / 2] = 1

    return disk


def Diamond(n_points):
    """
    Diamond : generate a diamond

    Author : Francois Leroux
    Contact : francois.leroux.pro@gmail.com
    Status : in progress
    Last update : 2024.04.05, Brest
    Comments :

    Inputs : MANDATORY : nPointsX {int} : number of sampling points in the X direction

    Outputs : a diamond
    """

    diamond = np.zeros((n_points, n_points))

    for k in range(n_points // 2 + 1):

        diamond[k, n_points // 2 - k : n_points // 2 + 1 + k] = 1
        diamond[n_points - k - 1, n_points // 2 - k : n_points // 2 + 1 + k] = 1

    return diamond


def Sun(n_points, *, width=0):
    """
    Sun : generate a Double cross (diagonal and straight)

    Author : Francois Leroux
    Contact : francois.leroux.pro@gmail.com
    Status : in progress
    Last update : 2024.04.05, Brest
    Comments : not working

    Inputs : MANDATORY : nPointsX {int} : number of sampling points in the X direction

    Outputs : a diamond
    """

    sun = np.zeros((n_points, n_points))
    for k in range(n_points):
        sun[k - width : k + width + 1, k - width : k + width + 1] = 1
        sun[k - width : k + width + 1, -(k - width) : -(k + width + 1)] = 1

    sun[n_points // 2 - width : n_points // 2 + width + 1, :] = 1
    sun[:, n_points // 2 - width : n_points // 2 + width + 1] = 1

    return sun


def Braille(letter):
    """
    Braille : generate the patern of a given letter in the Braille Alphabet,
              over a 3-by-2 pixels support

    Author : Francois Leroux
    Contact : francois.leroux.pro@gmail.com
    Status : in progress
    Last update : 2024.06.10, Brest
    Comments :

    Inputs : MANDATORY : letter {string} : the letter to return in Braille, in CAPITAL

    Outputs : the patern corresponding to letter in the Braille Alphabet,
              over a 3-by-2 pixels support
    """

    braille_letter = np.zeros((3, 2))

    if letter == "A":
        braille_letter[0, 0] = 1

    elif letter == "B":
        braille_letter[0, 0] = 1
        braille_letter[1, 0] = 1

    elif letter == "C":
        braille_letter[0, 0] = 1
        braille_letter[0, 1] = 1

    elif letter == "D":
        braille_letter[0, 0] = 1
        braille_letter[0, 1] = 1
        braille_letter[1, 1] = 1

    elif letter == "E":
        braille_letter[0, 0] = 1
        braille_letter[1, 1] = 1

    elif letter == "F":
        braille_letter[0, 0] = 1
        braille_letter[0, 1] = 1
        braille_letter[1, 0] = 1

    elif letter == "G":
        braille_letter[0, 0] = 1
        braille_letter[0, 1] = 1
        braille_letter[1, 0] = 1
        braille_letter[1, 1] = 1

    elif letter == "H":
        braille_letter[0, 0] = 1
        braille_letter[1, 0] = 1
        braille_letter[1, 1] = 1

    elif letter == "I":
        braille_letter[0, 1] = 1
        braille_letter[1, 0] = 1

    elif letter == "J":
        braille_letter[0, 1] = 1
        braille_letter[1, 0] = 1
        braille_letter[1, 1] = 1

    elif letter == "K":
        braille_letter[0, 0] = 1
        braille_letter[2, 0] = 1

    elif letter == "L":
        braille_letter[0, 0] = 1
        braille_letter[1, 0] = 1
        braille_letter[2, 0] = 1

    elif letter == "M":
        braille_letter[0, 0] = 1
        braille_letter[2, 0] = 1
        braille_letter[0, 1] = 1

    elif letter == "N":
        braille_letter[0, 0] = 1
        braille_letter[2, 0] = 1
        braille_letter[0, 1] = 1
        braille_letter[1, 1] = 1

    elif letter == "O":
        braille_letter[0, 0] = 1
        braille_letter[2, 0] = 1
        braille_letter[1, 1] = 1

    elif letter == "P":
        braille_letter[0, 0] = 1
        braille_letter[2, 0] = 1
        braille_letter[1, 0] = 1
        braille_letter[0, 1] = 1

    elif letter == "Q":
        braille_letter[0, 0] = 1
        braille_letter[2, 0] = 1
        braille_letter[1, 0] = 1
        braille_letter[0, 1] = 1
        braille_letter[1, 1] = 1

    elif letter == "R":
        braille_letter[0, 0] = 1
        braille_letter[2, 0] = 1
        braille_letter[1, 0] = 1
        braille_letter[1, 1] = 1

    elif letter == "S":
        braille_letter[0, 1] = 1
        braille_letter[2, 0] = 1
        braille_letter[1, 0] = 1

    elif letter == "T":
        braille_letter[0, 1] = 1
        braille_letter[2, 0] = 1
        braille_letter[1, 0] = 1
        braille_letter[1, 1] = 1

    elif letter == "U":
        braille_letter[0, 0] = 1
        braille_letter[2, 0] = 1
        braille_letter[2, 1] = 1

    elif letter == "V":
        braille_letter[0, 0] = 1
        braille_letter[1, 0] = 1
        braille_letter[2, 0] = 1
        braille_letter[2, 1] = 1

    elif letter == "W":
        braille_letter[1, 1] = 1
        braille_letter[1, 0] = 1
        braille_letter[2, 1] = 1
        braille_letter[0, 1] = 1

    elif letter == "X":
        braille_letter[0, 1] = 1
        braille_letter[0, 0] = 1
        braille_letter[2, 1] = 1
        braille_letter[2, 0] = 1

    elif letter == "Y":
        braille_letter[0, 1] = 1
        braille_letter[0, 0] = 1
        braille_letter[2, 1] = 1
        braille_letter[2, 0] = 1
        braille_letter[1, 1] = 1

    elif letter == "Z":
        braille_letter[1, 1] = 1
        braille_letter[0, 0] = 1
        braille_letter[2, 1] = 1
        braille_letter[2, 0] = 1

    return braille_letter
