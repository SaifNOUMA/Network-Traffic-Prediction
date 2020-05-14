import sys, os
sys.path.insert(1, "/home/senouma/stage2020/saif/version6/")
#%%

from fr_utils.utils import *

#%% Hyper Parameters

look_back, timestep = 20, 60
validation_split = 0.8  # Percentage of train set from the whole dataset
# look_back = 64          # Number of timesteps that will be fed to the network in order to predict the future timesteps
# look_ahead = 5         # Number of timesteps to be predicted
delete_percentage = {1:4000, 60: 100}
time = {1 : "second", 60: "minute"}

#%%
path_result = "results/svr/timestep_%d/lb=%d/"%(timestep, look_back)

if os.path.isdir("results/svr/timestep_%d"%timestep) == 0 :
    os.system("mkdir results/svr/timestep_%d"%(timestep))
if os.path.isdir(path_result)==0:
    os.system("mkdir "+path_result)
else :
    if len(glob(path_result+'*'))==2:
        exit()
    else:
        os.system("rm "+path_result+"*")

#%%
series = read_data(path="data/fdata_Timestep_%d"%timestep)
times  = range(len(series))
# plot_series(times, series, ylabel="Packets / "+time[timestep])

#%%
del_percentage = delete_percentage[timestep]
times , series = times[:-del_percentage], series[:-del_percentage]
# plot_series(times, series, ylabel="Packets / minute")

#%%

series_mean, series_std = series.mean() , series.std()
series = preprocessing.scale(series).reshape(len(series), 1)

train , test = train_test_split(series, validation_split)

y_preds = []
y_tests = []
errors = []

for look_ahead in range(1,21):
    train_x, train_y = create_datasets(train, look_back, look_ahead)
    test_x, test_y = create_datasets(test, look_back, look_ahead)
    train_y , test_y = train_y[:,look_ahead-1], test_y[:,look_ahead-1]
    # reshape the data to match Keras LSTM input gate [samples, time steps, features]
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1]))
    train_y = np.reshape(train_y, (train_y.shape[0], train_y.shape[1]))

    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1]))
    test_y = np.reshape(test_y, (test_y.shape[0], test_y.shape[1]))

    #%%

    clf = SVR(C=1.0, epsilon=0.01, gamma="auto")
    clf.fit(train_x, train_y.ravel())

    #%%
    pred_train = clf.predict(train_x)
    pred_test = clf.predict(test_x)

    pred_train = reverse_scale(pred_train, series_mean, series_std)
    pred_test = reverse_scale(pred_test, series_mean, series_std)
    test_y = reverse_scale(test_y, series_mean, series_std)
    train_y = reverse_scale(train_y, series_mean, series_std)

    test_y  = test_y.reshape(len(test_y))
    train_y = train_y.reshape(len(train_y))

    score = mean_absolute_percentage(test_y, pred_test)
    print("Mean Absolute Percentage Error: %f"%score)

    # plot_1_error(pred_test,test_y, score, path=path_result+"pred_error_%d"%look_ahead)
    errors.append(score)
    y_preds.append([pred_train,pred_test])
    y_tests.append([train_y,test_y])

#%%


columns=["look_back", "look_ahead", "neuron1", "neuron2", "neuron3"] + ["MAPE: %d Step"%i for i in range(1,21)]
values=[look_back, None, None, None, None]+ errors

dic_results = {column:value for column,value in zip(columns[:len(values)],values)}

df = pd.DataFrame(columns=columns)
df = df.append(dic_results, ignore_index=True)
df.to_csv(path_result+"metrics_comparison.csv",index=False)

dic = {"pred" : y_preds, "real" : y_tests}
path_file = path_result+"values.pickle"


with open(path_file, 'wb') as f:
    pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)

