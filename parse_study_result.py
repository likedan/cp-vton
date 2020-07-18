import csv
import pandas as pd
Right = {"(2)", "(3157)", "(4307)", "(4656)", "(998)", "(6047)", "(1170)", "(2436)", "(1786)", "(3287)", "(33)", "(58)", "(96)", "(148)", "(2393)", "(431)", "(1330)", "(1442)", "(1554)"}
right_title = set(["Which image looks more photo realistic? " + r for r in Right])

correct = 0
wrong = 0
draw = 0
with open('user_study2.csv') as csv_file:
    df = pd.read_csv(csv_file)
    for i, col in enumerate(df.items()):
        is_right = False
        if col[0] in right_title:
            is_right = True

        for j in range(40):
            try:
                value = col[1][j]
                print(value)
                if value == "Right Image":
                    if is_right:
                        correct += 1
                    else:
                        wrong += 1
                elif value == "Left Image":
                    if not is_right:
                        correct += 1
                    else:
                        wrong += 1
                elif value == "A Draw":
                    draw += 1
            except:
                continue

print("correct", correct, "wrong", wrong, "draw", draw)