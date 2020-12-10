import pandas as pd
import pprint
pd.set_option('display.max_rows', 5)
df = pd.read_csv("pact-2014-runtimes.csv")
# print(df)
# kset = set(df["kernel"])
# print(kset)
# for kname in kset:
# print(df["kernel"][0])
pp = pprint.PrettyPrinter(indent=4)

def set_min(temp, threads):
    s = [1,2,4,8,16,32]
    for i,v in enumerate(s):
        if v == threads:
            temp[i] = 1
        else:
            temp[i] = 0

def parse_seq(seq):
    l = seq.split()[1:]
    l[-1] = l[-1][:-1]
    for i, v in enumerate(l):
        l[i] = int(v)
    print(l)
    return l

name = ""
seq = []
min_time = -1
inp = dict()
res = dict()
GPU = ["runtime_Fermi", "runtime_Kepler", "runtime_Cypress", "runtime_Tahiti"]
temp = [0,0,0,0,0,0]
for gpu in GPU:
    for key, value in df["kernel"].items():
        if value != name:
            if min_time > 0:
                if gpu not in res:
                    res[gpu] = list()
                    inp[gpu] = list()
                inp[gpu].append(parse_seq(df["seq"][key]))
                print("set", gpu, name, temp)
                if name in res[gpu]:
                    print("ERROR!!!!!")
                res[gpu].append(temp.copy())
            name = value
            min_time = df[gpu][key]
            set_min(temp, df["cf"][key])
        elif df[gpu][key] < min_time:
            min_time = df[gpu][key]
            set_min(temp, df["cf"][key])
            print(gpu, value, df["cf"][key], df[gpu][key], temp)
    if gpu not in res:
        res[gpu] = list()
        inp[gpu] = list()
    inp[gpu].append(parse_seq(df["seq"][key]))
    print("set", gpu, name, temp)
    if name in res[gpu]:
        print("ERROR!!!!!")
    res[gpu].append(temp.copy())
    min_time = -1
pp.pprint(res)
print(inp)


for gpu, val in res.items():
    data = {'input': inp[gpu], 'output': val}
    write_df = pd.DataFrame(data)
    write_df.to_csv(gpu+'.csv')