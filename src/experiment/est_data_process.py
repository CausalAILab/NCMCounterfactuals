import argparse
import glob
import json
import os
import pandas as pd

from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description="Estimation Experiment Data Processor")
parser.add_argument('dir', help="directory of the experiment")
args = parser.parse_args()

d = args.dir

records = []
for entry in tqdm(glob.glob('{}/*'.format(d))):
    try:
        record = {}
        record['key'] = os.path.basename(entry)
        record.update(map(lambda t: tuple(t.split('=')), record['key'].split('-')))
        if len(record) < 3:
            continue

        with open(f'{entry}/results.json') as file:
            old_results = json.load(file)
            record.update(old_results)
        records.append(record)
    except Exception as e:
        print(entry, e)

df = pd.DataFrame(records)
os.makedirs('{}/results'.format(d), exist_ok=True)
df.to_csv('{}/results/est_data.csv'.format(d))
