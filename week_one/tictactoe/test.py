from tictactoe import minimax

X = "X"
O = "O"
EMPTY = None


def test_terminal():
    i = minimax([[EMPTY, X, EMPTY], [EMPTY, O, EMPTY], [EMPTY, X, O]])
    print(i)


test_terminal()
