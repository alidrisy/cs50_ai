from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(
    Or(AKnight, AKnave),
    Not(And(AKnight, AKnave)),
    Implication(AKnight, And(AKnight, AKnave)),
    Implication(AKnave, Not(And(AKnight, AKnave))),
)
# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(
    Or(AKnave, BKnave),
    Not(And(BKnave, AKnave)),
    Implication(BKnave, And(BKnave, AKnave)),
    Implication(AKnave, Not(And(BKnave, AKnave))),
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
knowledge2 = And(
    Or(AKnave, BKnave),
    Or(AKnight, BKnight),
    Not(And(AKnight, BKnight)),
    Not(And(AKnave, BKnave)),
    Implication(AKnight, And(AKnight, BKnight)),
    Implication(AKnave, Not(And(AKnight, BKnight))),
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
knowledge3 = And(
    Or(
        And(
            AKnight, BKnight, CKnave
        ),  # A is a knight, B tells the truth (A said "I am a knave", C is a knave)
        And(
            AKnave, BKnave, CKnight
        ),  # A is a knave, B lies (A did not say "I am a knave", C is a knight)
        And(
            AKnave, BKnight, CKnave
        ),  # A is a knave, B tells the truth (A said "I am a knave", C is a knave)
        And(
            AKnight, BKnave, CKnight
        ),  # A is a knight, B lies (A did not say "I am a knave", C is a knight)
    ),
    Implication(CKnight, And(AKnight, BKnave)),
    Implication(BKnight, And(AKnight, CKnave)),
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3),
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
