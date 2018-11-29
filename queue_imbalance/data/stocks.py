import os


def main():
    list_of_files = os.listdir(os.path.join('INDEX'))
    list_of_stocks = []
    for f in list_of_files:
        if '.csv' in f:
            s = f.split('_')
            list_of_stocks.append(s[1])
    print(set(list_of_stocks))
    return set(list_of_stocks)


if __name__ == '__main__':
    main()