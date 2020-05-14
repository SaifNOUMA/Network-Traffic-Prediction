from fr_utils.utils import *


#%%

columns=["algorithm", "look_back", "look_ahead", "neuron1", "neuron2", "neuron3"] + ["MAPE: %d Step"%i for i in range(1,21)]
dic_results = {column:None for column in columns}

df = pd.DataFrame(columns=columns)
df.to_csv("results/benchmark_mape.csv",index=False)
dir_path="results/"

#%%
for path in glob(dir_path+"*"):
    for path1 in glob(path+"/*"):
        if len(glob(path1+"/*")) is not 2:
            # %%  Complete the missing combinations that haven't been executed in the server.
            if "lstm" in path1:
                x = path1.split("/")[-1]
                lb, la, ne1, ne2, ne3 = [int(i.split("=")[-1]) for i in x.split("_")]
                os.system("python lstm.py %d %d %d %d %d" % (lb, la, ne1, ne2, ne3))
            else:
                x = path1.split("/")[-1]
                lb, la, ne1, ne2, ne3 = [int(i.split("=")[-1]) for i in x.split("_")]
                os.system("python gru.py %d %d %d %d %d" % (lb, la, ne1, ne2, ne3))

        ds = pd.read_csv(path1+"/metrics_comparison.csv")
        values = [path.split("/")[-1]] + list(ds.values[0,:len(columns)-1])
        dic_results = {column: value for column, value in zip(columns, values)}
        df = pd.read_csv("results/benchmark_mape.csv")
        df = df.append(dic_results, ignore_index=True)
        df.to_csv("results/benchmark_mape.csv", index=False)


#%%  Complete the missing combinations that haven't been executed in the server.
import os
for path in glob(dir_path+"*"):
    for path1 in glob(path+"/*"):
        if len(glob(path1+"/*")) is not 2:
            if "lstm" in path1:
                x = path1.split("/")[-1]
                lb, la, ne1, ne2, ne3 = [int(i.split("=")[-1]) for i in x.split("_")]
                os.system("python lstm.py %d %d %d %d %d"%(lb, la, ne1, ne2, ne3))
            else:
                x = path1.split("/")[-1]
                lb, la, ne1, ne2, ne3 = [int(i.split("=")[-1]) for i in x.split("_")]
                os.system("python gru.py %d %d %d %d %d" % (lb, la, ne1, ne2, ne3))


