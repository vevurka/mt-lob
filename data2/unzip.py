import argparse
import os
import subprocess


def main(file_name, sub_dir='data', password=None):
    pass_opt = []
    if password:
        pass_opt = ['-P', 'secret']

    subprocess.call(['unzip'] + pass_opt + [file_name] + ['-d', sub_dir])

    dirs = os.listdir(sub_dir)
    for d in dirs:
        files = os.listdir(os.path.join(sub_dir, d))

        for f in files:
            if 'zip' in f:
                subprocess.call(['unzip'] + pass_opt +
                                [os.path.join(sub_dir, d, f),
                                 '-d', os.path.join(sub_dir, d) + '/'])
                split = f.split('.')
                subprocess.call(
                    ['mv', os.path.join(sub_dir, d, 'OrderBookSnapshots.csv'),
                     os.path.join(sub_dir) + '/OrderBookSnapshots_{}_{}.csv'.format(split[2], d)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unzip data file for my master thesis')
    parser.add_argument('file_name',  type=str,
                        help='')
    parser.add_argument('--password', type=str,
                        help='')
    parser.add_argument('--sub_dir',  type=str, default='data', help='')

    args = parser.parse_args()
    main(args.file_name, sub_dir=args.sub_dir, password=args.password)
