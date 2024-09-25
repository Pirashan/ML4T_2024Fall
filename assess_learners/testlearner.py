import math
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
from datetime import datetime

def get_sample(x, y):
    if len(y.shape) == 1:
        y = np.reshape(y, (-1, 1))
    data = np.append(x, y, axis=1)
    obs = data.shape[0]
    rand_ind = np.random.randint(obs, size=obs)
    new_data = np.zeros(data.shape)
    for row in range(0, obs):
        new_data[row] = data[rand_ind[row]]
    new_x = new_data[:, 0:data.shape[1]-1]
    new_y = new_data[:, -1]
    return new_x, new_y

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    data = np.genfromtxt(sys.argv[1], delimiter=',')
    data = data[1:, 1:]

    # 60% training, 40% testing
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    # Experiment 1: DTLearner testing
    rmse_in_sample = []
    rmse_out_sample = []
    for i in range(1, 51):
        learner = dt.DTLearner(leaf_size=i, verbose=False)
        learner.add_evidence(train_x, train_y)

        pred_y = learner.query(train_x)
        rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        rmse_in_sample.append(rmse)

        pred_y2 = learner.query(test_x)
        rmse2 = math.sqrt(((test_y - pred_y2) ** 2).sum() / test_y.shape[0])
        rmse_out_sample.append(rmse2)

    plt.plot(rmse_in_sample)
    plt.plot(rmse_out_sample)
    plt.title("RMSE vs Leaf Size for DTLearner")
    plt.xlabel("Leaf Size")
    plt.xlim(1, 50)
    plt.ylabel("RMSE")
    plt.ylim(0, 0.01)
    plt.legend(["In-Sample", "Out-Sample"])
    plt.savefig("images/Experiment1.png")
    plt.close("all")

    # Experiment 2: BagLearner testing using DTLearner
    rmse_in_sample = []
    rmse_out_sample = []
    for i in range(1, 51):
        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": i}, bags=20, boost=False, verbose=False)
        learner.add_evidence(train_x, train_y)

        pred_y = learner.query(train_x)
        rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        rmse_in_sample.append(rmse)

        pred_y2 = learner.query(test_x)
        rmse2 = math.sqrt(((test_y - pred_y2) ** 2).sum() / test_y.shape[0])
        rmse_out_sample.append(rmse2)

    plt.plot(rmse_in_sample)
    plt.plot(rmse_out_sample)
    plt.title("RMSE vs Leaf Size for BagLearner with 20 Bags")
    plt.xlabel("Leaf Size")
    plt.xlim(1, 50)
    plt.ylabel("RMSE")
    plt.ylim(0, 0.01)
    plt.legend(["In-Sample", "Out-Sample"])
    plt.savefig("images/Experiment2.png")
    plt.close("all")

    # Experiment 3: DTLearner vs RTLearner
    # a) measuring training time
    time_dt = []
    time_rt = []
    # DTlearner and train it
    for i in range(1, 51):
        start = time.time()
        learner = dt.DTLearner(leaf_size=i, verbose=False)
        learner.add_evidence(train_x, train_y)
        end = time.time()
        time_dt.append(end - start)
    # RTlearner and train it
    for j in range(1, 51):
        start2 = time.time()
        learner2 = rt.RTLearner(leaf_size=j, verbose=False)
        learner2.add_evidence(train_x, train_y)
        end2 = time.time()
        time_rt.append(end2 - start2)

    plt.plot(time_dt)
    plt.plot(time_rt)
    plt.title("Training Time for DTLearner vs RTLearner")
    plt.xlabel("Leaf Size")
    plt.xlim(1, 50)
    plt.ylabel("Time (s)")
    plt.ylim(0, 0.1)
    plt.legend(["DTLearner", "RTLearner"])
    plt.savefig("images/Experiment3a.png")
    plt.close("all")

    # b) measuring Mean Absolute Error (MAE) for both
    mae_dt = []
    mae_rt = []
    learners = [[], []]
    number_trials = 50
    for i in range(number_trials):
        learners[0].append(dt.DTLearner(leaf_size=50))
        learners[1].append(rt.RTLearner(leaf_size=50))

    for i in range(number_trials):
        trial_train_x, trial_train_y = get_sample(train_x, train_y)
        # DT
        dt_now = datetime.now()
        learners[0][i].add_evidence(trial_train_x, trial_train_y)
        dt_later = datetime.now()
        dt_pred_y = learners[0][i].query(test_x)
        # mae
        mae_dt_temp = float(np.mean(np.abs(test_y - dt_pred_y)))
        mae_dt.append(mae_dt_temp)

        # RT
        rt_now = datetime.now()
        learners[1][i].add_evidence(trial_train_x, trial_train_y)
        rt_later = datetime.now()
        rt_pred_y = learners[1][i].query(test_x)
        # mae
        mae_rt_temp = float(abs(test_y - rt_pred_y).sum() / test_y.shape[0])
        mae_rt.append(mae_rt_temp)

    # plotting the figure
    plt.plot(mae_dt)
    plt.plot(mae_rt)
    plt.title("MAE for Training DTLearner vs RTLearner")
    plt.xlabel("Leaf Size")
    plt.xlim(1, 50)
    plt.ylabel("MAE")
    plt.ylim(0, 0.01)
    plt.legend(["DTLearner", "RTLearner"])
    plt.savefig("images/Experiment3b.png")
    plt.close("all")