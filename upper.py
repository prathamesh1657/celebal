def upper_triangle(rows):
    for i in range(rows):
        for j in range(rows - i):
            print("* ", end="")
        print()

upper_triangle(5)