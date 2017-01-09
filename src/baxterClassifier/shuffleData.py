import random


def main():
    # SHUFFLE THE LINES!
    with open('data/data.csv', 'r') as source:
        data = [(random.random(), line) for line in source]

    data.sort()
    test_data_file = open("data/test_data.csv", "w")
    test_threshold = int(len(data) * 0.9)

    with open('data/train_data.csv', 'w') as target:
        count = 0
        for _, line in data:
            count = count + 1
            if count < test_threshold:
                target.write(line)
            else:
                test_data_file.write(line)

    test_data_file.close()

if __name__ == '__main__':
    main()
