import chess
from typeguard import typechecked

from src.my_torch_analyzer.chessboard import STATE_LABELS, Chessboard


@typechecked
def expected_label(board: chess.Board) -> str:
    """Label of a position (colour = side giving the check)."""
    if not board.is_check():
        return "Nothing"
    attacker = "White" if board.turn == chess.BLACK else "Black"
    return f"{'Checkmate' if board.is_checkmate() else 'Check'} {attacker}"

@typechecked
def parse_line(line: str, check_consistency: bool = False):
    """Validate one dataset line and return (fen, result)."""
    line = line.strip()
    parts = line.split()
    if len(parts) < 6:
        raise ValueError(f"FEN needs 6 fields, got {len(parts)}: {line!r}")

    fen = " ".join(parts[:6])
    result = " ".join(parts[6:]) or None

    board = chess.Board(fen, chess960=True)
    if not board.is_valid():
        raise ValueError(f"Illegal position: {fen!r}")

    if result is not None:
        if STATE_LABELS.get(result.lower()) is None:
            raise ValueError(f"Unknown result {result!r}, expected one of {sorted(STATE_LABELS)}")
        if check_consistency and result != expected_label(board):
            raise ValueError(f"Result {result!r} does not match position (is {expected_label(board)!r})")

    return Chessboard(fen, result)

@typechecked
def fen_parser(filename: str) -> list[Chessboard]:
    data = []
    with open(filename) as f:
        for line in f:
            data.append(parse_line(line))
    return data
