def pyramid(rows):
    for i in range(rows):
        for j in range(rows - i - 1):
            print("  ", end="")
        for k in range(2 * i + 1):
            print("* ", end="")
        print()

pyramid(5)