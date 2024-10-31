"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    if initial_state() == board:
        return X
    num_x = 0
    num_o = 0

    for i in board:
        for n in i:
            if n == X:
                num_x += 1
            if n == O:
                num_o += 1
    if num_x <= num_o:
        return X
    return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                actions.add((i, j))
    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i, j = action
    if board[i][j] is not EMPTY:
        raise ValueError("Invalid action: Cell is already filled.")

    # Create a deep copy of the board and make the move
    new_board = copy.deepcopy(board)
    new_board[i][j] = player(board)
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    if not terminal(board):
        return None
    i = utility(board)
    if i == 1:
        return X
    if i == -1:
        return O
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if all(EMPTY not in i for i in board) or utility(board) != 0:
        return True
    return False


def utility(board):
    # Check rows for a win
    for row in board:
        if row[0] == row[1] == row[2] and row[0] is not None:
            return 1 if row[0] == "X" else -1

    # Check columns for a win
    for col in range(3):
        if (
            board[0][col] == board[1][col] == board[2][col]
            and board[0][col] is not None
        ):
            return 1 if board[0][col] == "X" else -1

    # Check diagonals for a win
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] is not None:
        return 1 if board[0][0] == "X" else -1
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] is not None:
        return 1 if board[0][2] == "X" else -1

    # If no winner, it's a tie
    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    def max_value(board, alpha, beta):
        if terminal(board):
            return utility(board), None

        value = float("-inf")
        best_action = None

        for action in actions(board):
            # Get minimum value for this action
            min_val, _ = min_value(result(board, action), alpha, beta)

            # Update best value and action if we found a better move
            if min_val > value:
                value = min_val
                best_action = action

            # Alpha-beta pruning
            alpha = max(alpha, value)
            if alpha >= beta:
                break

        return value, best_action

    def min_value(board, alpha, beta):
        if terminal(board):
            return utility(board), None

        value = float("inf")
        best_action = None

        for action in actions(board):
            # Get maximum value for this action
            max_val, _ = max_value(result(board, action), alpha, beta)

            # Update best value and action if we found a better move
            if max_val < value:
                value = max_val
                best_action = action

            # Alpha-beta pruning
            beta = min(beta, value)
            if alpha >= beta:
                break

        return value, best_action

    # Initialize alpha and beta values for pruning
    alpha = float("-inf")
    beta = float("inf")

    # Get the optimal move based on current player
    current_player = player(board)
    if current_player == X:
        _, best_move = max_value(board, alpha, beta)
    else:
        _, best_move = min_value(board, alpha, beta)

    return best_move
