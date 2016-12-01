import random


def main():
    # SHUFFLE THE LINES!
    with open('data.csv', 'r') as source:
        data = [(random.random(), line) for line in source]
    data.sort()

    with open('final_data.csv', 'w') as target:
        for _, line in data:
            target.write(line)


if __name__ == '__main__':
    main()
