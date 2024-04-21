
if __name__ == '__main__':

    # unpacking the tuple using zip
    pitchers = [
        ('Nolan Ryan', 'TEX'),
        ('Roger Clemens', 'BOS'),
        ('Curt Schilling', 'PHI'),
        ('Randy Johnson', 'ARI')
    ]

    # unpacking the tuple using zip
    names, teams = zip(*pitchers)

    # work with dictionary (sort by first letter)
    fruits = ['apple', 'banana', 'cherry', 'pineapple', 'mango']

    by_letter = {}

    for fruit in fruits:
        letter = fruit[0]
        if letter not in by_letter:
            by_letter[letter] = [fruit]
        else:
            by_letter[letter].append(fruit)

    print(by_letter)

    # using setdefault
    by_letter = {}

    for fruit in fruits:
        by_letter.setdefault(fruit[0], []).append(fruit)

    print(by_letter)

    # using defaultdict
    from collections import defaultdict

    by_letter = defaultdict(list)

    for fruit in fruits:
        by_letter[fruit[0]].append(fruit)

    print(by_letter)

    # working with set
    a = {1, 2, 3, 4, 5}
    b = {3, 4, 5, 6, 7, 8}

    # union
    print(a | b)
    print(a.union(b))

    # intersection
    print(a & b)
    print(a.intersection(b))

    # difference
    print(a - b)
    print(a.difference(b))

    # symmetric difference
    print(a ^ b)
    print(a.symmetric_difference(b))

    # check if a is subset of b
    print(a.issubset(b))
    print(a <= b)

    # check if a is superset of b
    print(a.issuperset(b))
    print(a >= b)

    # check if a is proper subset of b
    print(a < b)
    print(a.issubset(b))

    # check if a is proper superset of b
    print(a > b)
    print(a.issuperset(b))

    # check if a and b are disjoint
    print(a.isdisjoint(b))

    # list comprehension
    strings = ['a', 'as', 'bat', 'car', 'dove', 'python']
    print([x.upper() for x in strings if len(x) > 2])

    # set comprehension
    print({len(x) for x in strings})

    # dictionary comprehension
    unique_lengths = {len(x) for x in strings}

    loc_mapping = {val: index for index, val in enumerate(strings)}

    print(loc_mapping)

    # nested list comprehension
    all_data = [['John', 'Emily', 'Michael', 'Mary', 'Steven'],
                ['Maria', 'Juan', 'Javier', 'Natalia', 'Pilar']]

    names_of_interest = []

    for names in all_data:
        enough_es = [name for name in names if name.count('e') >= 2]
        names_of_interest.extend(enough_es)

    print(names_of_interest)

    # nested list comprehension
    result = [name for names in all_data for name in names if name.count('e') >= 2]

    # function and partial function
    def add_numbers(x, y):
        return x + y
    # fixing the value of x using lambda function
    add_five = lambda y: add_numbers(5, y)

    # using partial function, we can fix the value of one or more arguments
    from functools import partial
    add_five = partial(add_numbers, 5)
    print(add_five(10))

    # generator and itertools

    # generator
    def squares(n=10):
        for i in range(1, n + 1):
            yield i ** 2

    gen = squares()
    print(gen)

    for x in gen:
        print(x)

    # itertools
    import itertools

    first_letter = lambda x: x[0]

    names = ['Alan', 'Adam', 'Wes', 'Will', 'Albert', 'Steven']

    for letter, names in itertools.groupby(names, first_letter):
        print(letter, list(names))


