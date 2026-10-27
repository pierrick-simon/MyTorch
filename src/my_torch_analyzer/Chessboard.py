from typeguard import typechecked
from enum import Enum

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

    @typechecked
    def getPiece(letter: str):
        tmp = letter.lower()
        match tmp[0]:
            case 'k':
                return ChessPieces.KING
            case 'q':
                return ChessPieces.QUEEN
            case 'r':
                return ChessPieces.ROOK
            case 'b':
                return ChessPieces.BISHOP
            case 'n':
                return ChessPieces.KNIGHT
            case _:
                return ChessPieces.PAWN


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
        self.piece = ChessPieces.getPiece(letter).value
        self.list[self.piece + 6 * self.color] = 1

    @typechecked
    def getList(self) -> list[float]:
        return self.list

class Chessboard:
    @typechecked
    def __init__(self, fen: str):
        self.squares = []
        parts = fen.split()
        for letter in parts[Fen.PLACEMENT.value]:
            if (letter == "/"):
                continue
            elif (letter.isnumeric()):
                for _ in(0, int(letter), 1):
                    self.squares.append(Square())
            else:
                self.squares.append(Square(letter))

    @typechecked
    def getInput(self) -> list[float]:
        list = []
        for square in self.squares:
            list.append(square.getList())
        return list