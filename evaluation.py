import sys

PATTERNS = {
"white_line1": """    ---
    ---
    ---
--- --- --- ---
--- --- --- ---
--- -O- --- -R-
    -W-
    -W-
    -W-""",
"white_line2": """    ---
    ---
    ---
--- --- --- ---
--- --- --- ---
-G- --- -B- ---
    ---
    WWW
    ---""",
"white_cross": """    ---
    ---
    ---
--- --- --- ---
--- --- --- ---
--- -O- -B- -R-
    -W-
    WWW
    -W-""",
"white_corners": """    ---
    ---
    ---
--- --- --- ---
--- --- --- ---
GGG OOO BBB RRR
    WWW
    WWW
    WWW""",
"F2L": """    ---
    ---
    ---
--- --- --- ---
GGG OOO BBB ---
GGG OOO BBB ---
    WWW
    WWW
    WWW""",
"yellow_cross": """    -Y-
    YYY
    -Y-
GGG OOO BBB RRR
GGG OOO BBB RRR
GGG OOO BBB RRR
    WWW
    WWW
    WWW""",
"all": """    YYY
    YYY
    YYY
GGG OOO BBB RRR
GGG OOO BBB RRR
GGG OOO BBB RRR
    WWW
    WWW
    WWW"""
}

def score_cube(c):
    scores = {
    #     "white_line1": 8,
    #     "white_line2": 8,
    #     "white_cross": 3,
    #     "white_corners": 2,
    #     "F2L": 1,
    #     "yellow_cross": 1,
        "all": 1
    }
    #
    # # iterate over all keys and values in scores
    for key, value in scores.items():
        scores[key] = face_matches(c, key) * value
    # total_score = 0
    # scores = {}
    # scores["white_line1"] = face_matches(c, "white_line1") * 8
    # scores["white_line2"] = face_matches(c, "white_line2") * 8
    # if scores["white_line1"] >= 5 * 8 and scores["white_line2"] >= 5 * 8:
    #     scores["white_cross"] = face_matches(c, "white_cross") * 3
    #     if scores["white_cross"] >= 9 * 3:
    #         scores["white_corners"] = face_matches(c, "white_corners") * 2
    #         if scores["white_corners"] >= 20:
    #             scores["F2L"] = face_matches(c, "F2L") * 5
    #             if scores["F2L"] >= 150:
    #                 scores["yellow_cross"] = face_matches(c, "yellow_cross") * 2
    #                 if scores["yellow_cross"] >= 95:
    #                     scores["all"] = face_matches(c, "all") * 2

    # print(c)

    # sum the scores
    total_score = sum(scores.values())
    print("\rScore: ", total_score, "Scores: ", scores, end="")
    sys.stdout.flush()
    return total_score


def face_matches(cube, kind):
    # print(pattern)
    return char_match_count(PATTERNS[kind], str(cube))


def char_match_count(pattern, cube_str):
    # count the number of matching characters between pattern and cube_str
    # if the number of matching characters is 9, then the white cross is complete
    count = 0
    for i in range(len(pattern)):
        # pattern[i] has to be W, Y, G, R, B, or O
        if pattern[i] == cube_str[i] and pattern[i] in "WYGBOR":
            count += 1
    return count
