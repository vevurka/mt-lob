import os
import subprocess

sub_dir = 'LOB'
subprocess.call(['unzip', 'LSEROB.2018.03.08.zip', '-d', sub_dir])

dirs = os.listdir(sub_dir)
for d in dirs:
    files = os.listdir(os.path.join(sub_dir, d))

    for f in files:
        if 'zip' in f:
            subprocess.call(['unzip',
                             os.path.join(sub_dir, d, f), '-d', os.path.join(sub_dir, d) + '/'])
            split = f.split('.')
            subprocess.call(
                ['mv', os.path.join(sub_dir, d, 'OrderBookSnapshots.csv'),
                 os.path.join(sub_dir) + '/OrderBookSnapshots_{}_{}.csv'.format(split[2], d)])
