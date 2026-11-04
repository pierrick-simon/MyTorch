from enum import Enum

from typeguard import typechecked


class Color(Enum):
    BLACK = 0
    WHITE = 1

    @typechecked
    def getColor(letter: str):
        if (letter.islower()):
            return Color.BLACK
        return Color.WHITE

class ChessPieces(Enum):
    KING = 0
    QUEEN = 1
    ROOK = 2
    BISHOP = 3
    KNIGHT = 4
    PAWN = 5

CHESS_PIECES_LABELS = {
    "k": ChessPieces.KING,
    "q": ChessPieces.QUEEN,
    "r": ChessPieces.ROOK,
    "b": ChessPieces.BISHOP,
    "n": ChessPieces.KNIGHT,
    "p": ChessPieces.PAWN,
}

@typechecked
def getPiece(letter: str) -> ChessPieces:
    return CHESS_PIECES_LABELS[letter.lower()]


class Fen(Enum):
    PLACEMENT = 0
    ACTIVECOLOR = 1
    CASTLING = 2
    PASSANT = 3
    HALFMOVE = 4
    FULLMOVE = 5

class State(Enum):
    NOTHING = 0
    CHECK_WHITE = 1
    CHECK_BLACK = 2
    CHECKMATE_WHITE = 3
    CHECKMATE_BLACK = 4

STATE_LABELS = {
    "nothing": State.NOTHING,
    "check white": State.CHECK_WHITE,
    "check black": State.CHECK_BLACK,
    "checkmate white": State.CHECKMATE_WHITE,
    "checkmate black": State.CHECKMATE_BLACK,
}

@typechecked
def getState(state: str) -> State:
    return STATE_LABELS[state.lower()]

@typechecked
def getStateStr(output: list[float]) -> str:
    index = output.index(max(output))
    return State(index).name.replace("_", " ").title()

Grid = list[list[str | None]]

DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
KNIGHT_JUMPS = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
NB_RAY_VALUES = 13
MAX_DISTANCE = 7
NB_INPUTS = 8 * NB_RAY_VALUES + 8 + 2 + 8 * 2 + 8 + 3
GRID_SIZE = 8


def _flag(condition: bool) -> float:
    return 1.0 if condition else 0.0

def _onBoard(row: int, column: int) -> bool:
    return 0 <= row < GRID_SIZE and 0 <= column < GRID_SIZE

def _isEnemy(letter: str, white: bool) -> bool:
    return letter.isupper() != white

def _pieceIndex(letter: str, white: bool) -> int:
    piece = CHESS_PIECES_LABELS[letter.lower()].value
    if _isEnemy(letter, white):
        piece += 6
    return piece

def _hasPiece(grid: Grid, row: int, column: int, letter: str) -> bool:
    return _onBoard(row, column) and grid[row][column] == letter

def _enemyKnight(white: bool) -> str:
    if white:
        return "n"
    return "N"

def _enemyPawn(white: bool) -> str:
    if white:
        return "p"
    return "P"

def _ownKing(white: bool) -> str:
    if white:
        return "K"
    return "k"

def _pawnSquares(row: int, column: int, white: bool) -> list[tuple[int, int]]:
    pawnRow = row
    if white:
        pawnRow -= 1
    else:
        pawnRow += 1
    return [(pawnRow, column - 1), (pawnRow, column + 1)]

def _parseBoard(placement: str) -> Grid:
    grid: Grid = [[None] * GRID_SIZE for _ in range(GRID_SIZE)]
    row = column = 0
    for letter in placement:
        if letter == "/":
            row += 1
            column = 0
        elif letter.isnumeric():
            column += int(letter)
        else:
            grid[row][column] = letter
            column += 1
    return grid

def _findKing(grid: Grid, white: bool) -> tuple[int, int]:
    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE):
            if grid[row][column] == _ownKing(white):
                return row, column
    raise ValueError("no king for the side to move")

def _withoutPiece(grid: Grid, row: int, column: int) -> Grid:
    copy = [line[:] for line in grid]
    copy[row][column] = None
    return copy

def _scanRay(grid: Grid, row: int, column: int, dr: int, dc: int) -> tuple[str | None, int]:
    distance = 0
    r, c = row + dr, column + dc
    while _onBoard(r, c):
        distance += 1
        if grid[r][c] is not None:
            return grid[r][c], distance
        r, c = r + dr, c + dc
    return None, distance

def _canCaptureAlong(letter: str, directionIndex: int, distance: int) -> bool:
    """Can this piece capture along a ray? Queen: both, rook: straight, bishop: diagonal, king: adjacent."""
    kind = letter.lower()
    straight = directionIndex % 2 == 0
    return (kind == "q"
            or (kind == "r" and straight)
            or (kind == "b" and not straight)
            or (kind == "k" and distance == 1))

def _attackedAlongRays(grid: Grid, row: int, column: int, white: bool) -> bool:
    for index, (dr, dc) in enumerate(DIRECTIONS):
        letter, distance = _scanRay(grid, row, column, dr, dc)
        if letter is not None and _isEnemy(letter, white) and _canCaptureAlong(letter, index, distance):
            return True
    return False

def _attackedByKnight(grid: Grid, row: int, column: int, white: bool) -> bool:
    knight = _enemyKnight(white)
    for dr, dc in KNIGHT_JUMPS:
        if _hasPiece(grid, row + dr, column + dc, knight):
            return True
    return False

def _attackedByPawn(grid: Grid, row: int, column: int, white: bool) -> bool:
    pawn = _enemyPawn(white)
    for r, c in _pawnSquares(row, column, white):
        if _hasPiece(grid, r, c, pawn):
            return True
    return False

def _isAttacked(grid: Grid, row: int, column: int, white: bool) -> bool:
    return (_attackedAlongRays(grid, row, column, white)
            or _attackedByKnight(grid, row, column, white)
            or _attackedByPawn(grid, row, column, white))

def _rayFeatures(grid: Grid, kingRow: int, kingColumn: int, white: bool) -> list[float]:
    features: list[float] = []
    for dr, dc in DIRECTIONS:
        letter, distance = _scanRay(grid, kingRow, kingColumn, dr, dc)
        ray = [0.0] * NB_RAY_VALUES
        if letter is not None:
            ray[_pieceIndex(letter, white)] = 1.0
        ray[NB_RAY_VALUES - 1] = distance / MAX_DISTANCE
        features.extend(ray)
    return features

def _knightFeatures(grid: Grid, kingRow: int, kingColumn: int, white: bool) -> list[float]:
    knight = _enemyKnight(white)
    features: list[float] = []
    for dr, dc in KNIGHT_JUMPS:
        features.append(_flag(_hasPiece(grid, kingRow + dr, kingColumn + dc, knight)))
    return features

def _pawnFeatures(grid: Grid, kingRow: int, kingColumn: int, white: bool) -> list[float]:
    pawn = _enemyPawn(white)
    features: list[float] = []
    for r, c in _pawnSquares(kingRow, kingColumn, white):
        features.append(_flag(_hasPiece(grid, r, c, pawn)))
    return features

def _neighbourFeatures(grid: Grid, kingRow: int, kingColumn: int, white: bool) -> list[float]:
    features: list[float] = []
    for dr, dc in DIRECTIONS:
        r, c = kingRow + dr, kingColumn + dc
        onBoard = _onBoard(r, c)
        features.append(_flag(not onBoard))
        features.append(_flag(onBoard and grid[r][c] is not None and not _isEnemy(grid[r][c], white)))
    return features

def _attackedNeighbourFeatures(grid: Grid, kingRow: int, kingColumn: int, white: bool) -> list[float]:
    emptied = _withoutPiece(grid, kingRow, kingColumn)
    features: list[float] = []
    for dr, dc in DIRECTIONS:
        features.append(_flag(_onBoard(kingRow + dr, kingColumn + dc) and _isAttacked(emptied, kingRow + dr, kingColumn + dc, white)))
    return features

def _kingFeatures(kingRow: int, kingColumn: int, white: bool) -> list[float]:
    return [kingRow / MAX_DISTANCE, kingColumn / MAX_DISTANCE, _flag(white)]

def _kingRelativeInput(placement: str, white: bool) -> list[float]:
    grid = _parseBoard(placement)
    row, column = _findKing(grid, white)
    return (_rayFeatures(grid, row, column, white)
            + _knightFeatures(grid, row, column, white)
            + _pawnFeatures(grid, row, column, white)
            + _neighbourFeatures(grid, row, column, white)
            + _attackedNeighbourFeatures(grid, row, column, white)
            + _kingFeatures(row, column, white))

class Chessboard:
    @typechecked
    def __init__(self, fen: str, state: str | None):
        parts = fen.split()
        activeColor = parts[Fen.ACTIVECOLOR.value]
        self.input: list[float] = _kingRelativeInput(parts[Fen.PLACEMENT.value], activeColor == "w")
        self.state = [0, 0, 0, 0, 0]
        self.has_state = False
        if (state != None):
            self.state[getState(state).value] = 1
            self.has_state = True

    @typechecked
    def getInput(self) -> list[float]:
        return self.input

    @typechecked
    def hasState(self) -> bool:
        return self.has_state

    @typechecked
    def getSate(self) -> list[float]:
        return self.state