
#%%
from fr_utils.utils import *

df  = pd.read_csv("results/benchmark_mape.csv")
df1 = df.sort_values(by=["MAPE: %d Step"%i for i in range(1,21)])
df1.to_csv("results/benchmark.csv", index=False)

#%%

df = pd.read_csv("results/benchmark.csv")

values = df[["algorithm","look_ahead", "MAPE: 1 Step"]].values


# Comparison of MAPE corresponding to the look_back hyperparameter :
paths = glob("results/*")
look_back = list(range(20,101,10))
lh, ne1 , ne2, ne3 = 20, 50, 50, 50
lstm_values, gru_values = [], []

for look_back in range(20,101,10):
    path1 = "/lb=%d_la=%d_ne1=%d_ne2=%d_ne=%d/metrics_comparison.csv"%(look_back,lh,ne1,ne2,ne3)
    for path0 in paths:
        path = path0 + path1
        if len(glob(path0+'/'+path1.split("/")[1]+"/*"))!=2:
            if "gru" in path:
                os.system("python gru.py %d %d %d %d %d" % (look_back, lh, ne1, ne2, ne3))
            elif "lstm" in path:
                os.system("python lstm.py %d %d %d %d %d" % (look_back, lh, ne1, ne2, ne3))

        if "gru" in path:
            ds_temp = pd.read_csv(path)
            gru_values.append(ds_temp[["MAPE: %d Step"%i for i in range(1,21)]].values[0,:])
        elif "lstm" in path:
            ds_temp = pd.read_csv(path)
            lstm_values.append(ds_temp[["MAPE: %d Step"%i for i in range(1,21)]].values[0,:])


#%%

lstm_values, gru_values = np.array(lstm_values), np.array(gru_values)
look_back = list(range(20,101,10))
for i in range(20):
    plt.plot(look_back, lstm_values[:,i], marker="^", label="LSTM")
    plt.plot(look_back, gru_values[:,i], marker="o", label="GRU")
    plt.xlabel("Look Back")
    plt.ylabel("MAPE for %d Ahead"%(i+1))
    plt.title("Comparison Between different DL Algos")
    plt.legend(loc=1, fontsize=8, framealpha=0.8)
    plt.savefig("results/analysis/comparison_N%d_lh.%d_ne1.%d_ne2.%d_ne3.%d.png"% (i, lh, ne1, ne2, ne3))
    plt.close()


#%%
path1 = "/lb=%d_la=%d_ne1=%d_ne2=%d_ne=%d/"
for  i in range(10,101,10):
    os.system("rm -rf results/gru/lb=%d_la=%d_ne1=%d_ne2=%d_ne=%d"%(i,50,50,50,50))
    os.system("python lstm.py %d %d %d %d %d" % (i, lh, ne1, ne2, ne3))


