import glob
import numpy as np
import pickle
from multiprocessing import Process


def get_features(file_list):
    feature = np.load(file_list[0]).reshape((1024, 1))
    for i in range(1, len(file_list)):
        feature_i = np.load(file_list[i]).reshape((1024, 1))
        feature = np.concatenate([feature, feature_i], axis=1)
    return feature


def compare_features(val_feature_folder,
                     all_feature_folder,
                     worker_id=0,
                     output_folder="",
                     num_workers=1,
                     step=4096,
                     threshold=0.95):
    black_list = []
    val_file_list = glob.glob("{}/*/*.npy".format(val_feature_folder))
    val_features = get_features(val_file_list)
    all_file_list = glob.glob("{}/*/*.npy".format(all_feature_folder))
    l = len(all_file_list)
    local_file_list = all_file_list[l * worker_id // num_workers:l * (
        worker_id + 1) // num_workers]
    l1 = len(local_file_list)
    for i in range(0, l1, step):
        step_files = local_file_list[i:i + step]
        features = get_features(step_files)
        sim = features @val_features.T
        max_sim = np.max(sim, axis=0)
        indexes = np.where(max_sim > threshold)
        black_list.extend(step_files[indexes])
    output_file = "{}/{}.pkl".format(output_folder, worker_id)
    with open(output_file, 'w') as f:
        pickle.dump(black_list, f)


def main():
    val_feature_folder = ""
    all_feature_folder = ""
    worker_num = 128
    p_list = []
    for i in range(worker_num):
        p = Process(
            target=compare_features,
            args=(val_feature_folder, all_feature_folder, i))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()


if __name__ == '__main__':
    main()
