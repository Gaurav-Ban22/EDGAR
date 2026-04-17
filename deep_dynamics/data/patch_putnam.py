import sys
import math

csv_path = 'Putnam_park2023_run2_1.csv'

with open(csv_path, 'r') as f:
    lines = f.readlines()

header = []
data_lines = []
for line in lines:
    if line.startswith('#'):
        header.append(line)
    elif len(line.strip()) > 0:
        data_lines.append([float(x) for x in line.strip().split(',')])

end_row = data_lines[-1]
start_row = None
for row in data_lines:
    if row[3] >= 5.0: # first row with vx >= 5
        start_row = row
        break

if start_row is None:
    start_row = data_lines[0]

N = 800 # 32 seconds at 0.04s

synthetic_lines = []
for i in range(1, N + 1):
    r = i / float(N)
    new_row = []
    for j in range(len(end_row)):
        if j == 0: # time
            new_row.append(end_row[0] + i * 0.04)
        else:
            val = end_row[j] + r * (start_row[j] - end_row[j])
            new_row.append(val)
    synthetic_lines.append(new_row)

data_lines.extend(synthetic_lines)

with open(csv_path, 'w') as f:
    f.writelines(header)
    for row in data_lines:
        f.write(','.join([f"{v:.8f}" for v in row]) + '\n')

print(f"Appended {N} synthetic rows to {csv_path} to close the loop.")
