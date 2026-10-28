from enum import Enum

from typeguard import typechecked


class Color(Enum):
    BLACK = 0
    WHITE = 1

    @typechecked
    def getColor(letter: str):
        if (letter.islower()):
            return Color.BLACK
        return Color.BLACK

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

class Square:
    @typechecked
    def __init__(self, letter: str = ""):
        self.list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if (letter == ""):
            return
        self.color = Color.getColor(letter).value
        self.piece = getPiece(letter).value
        self.list[self.piece + 6 * self.color] = 1

    @typechecked
    def getList(self) -> list[float]:
        return self.list

class State(Enum):
    NOTHING = 0
    CHECK = 1

STATE_LABELS = {
    "nothing": State.NOTHING,
    "check white": State.CHECK,
    "check black": State.CHECK,
    "checkmate white": State.CHECK,
    "checkmate black": State.CHECK,
}

@typechecked
def getState(state: str) -> State:
    return STATE_LABELS[state.lower()]

@typechecked
def getStateStr(output: list[float]) -> str:
    index = output.index(max(output))
    return State(index).name.capitalize()

class Chessboard:
    @typechecked
    def __init__(self, fen: str, state: str | None):
        self.squares: list[Square] = []
        parts = fen.split()
        for letter in parts[Fen.PLACEMENT.value]:
            if (letter == "/"):
                continue
            elif (letter.isnumeric()):
                for _ in range (0, int(letter), 1):
                    self.squares.append(Square())
            else:
                self.squares.append(Square(letter))
        self.extra: list[float] = []
        self.extra.append(1.0 if parts[Fen.ACTIVECOLOR.value] == "w" else 0.0)
        castling = parts[Fen.CASTLING.value]
        for letter in "KQkq":
            self.extra.append(1.0 if letter in castling else 0.0)
        passant = parts[Fen.PASSANT.value]
        for file in "abcdefgh":
            self.extra.append(1.0 if passant.startswith(file) else 0.0)
        self.extra.append(min(int(parts[Fen.HALFMOVE.value]), 100) / 100)
        self.extra.append(min(int(parts[Fen.FULLMOVE.value]), 200) / 200)
        self.state = [0, 0]
        self.has_state = False
        if (state != None):
            self.state[getState(state).value] = 1
            self.has_state = True

    @typechecked
    def getInput(self) -> list[float]:
        tmp: list[float] = []
        for square in self.squares:
            tmp.extend(square.getList())
        tmp.extend(self.extra)
        return tmp

    @typechecked
    def hasState(self) -> bool:
        return self.has_state

    @typechecked
    def getSate(self) -> list[float]:
        return self.state