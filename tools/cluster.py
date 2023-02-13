import numpy as np
import faiss
import os
from multiprocessing import Queue, Process


def save_result(feature, path_list, center_feature, output_path):
    if not path_list:
        return
    global CLUSTER_ID
    save_path = "{}/{}".format(output_path, CLUSTER_ID)
    CLUSTER_ID += 1
    os.makedirs(save_path, exist_ok=True)
    path_raw = ""
    for path_i in path_list:
        path_raw += "{}\n".format(
            path_i.replace("features", "image").replace("npy", "JPEG"))
    path_raw = path_raw[:-1]
    with open("{}/image_list.txt".format(save_path), "w") as f:
        f.write(path_raw)
    np.save("{}/features.npy".format(save_path), feature)
    np.save("{}/center.npy".format(save_path), center_feature)


def cluster(feature,
            path_list,
            level,
            total_level,
            total_num,
            cluster_per_level,
            center_feature=None,
            cluster_iter=20,
            output_path="cluster_output"):
    if level == total_level:
        save_result(feature, path_list, center_feature, output_path)
        return
    total_cluster_num = cluster_per_level**(level + 1)
    num_per_cluster = total_num // total_cluster_num
    current_num = feature.shape[0]
    current_cluster_num = current_num // num_per_cluster
    if current_cluster_num <= 1:
        cluster(feature, path_list, level + 1, total_level, total_num,
                cluster_per_level)
        return
    kmeans = faiss.Kmeans(
        feature.shape[1],
        current_cluster_num,
        niter=cluster_iter,
        verbose=True)
    kmeans.train(feature)
    result = kmeans.assign(feature)
    for i in range(current_cluster_num):
        feature_list_i = []
        path_list_i = []
        for j, result_j in enumerate(result[1]):
            if result_j == i:
                feature_list_i.append(feature[j].reshape(1, 1024))
                path_list_i.append(path_list[j])
        if not feature_list_i:
            continue
        feature_i = np.concatenate(feature_list_i, axis=0)
        center_feature = kmeans.centroids[i].reshape(1, 1024)
        if not i % 8:
            print("begin level{}, cluster{}".format(level + 1, i))
        cluster(
            feature_i,
            path_list_i,
            level + 1,
            total_level,
            total_num,
            cluster_per_level,
            center_feature=center_feature,
            cluster_iter=20,
            output_path=output_path)


def mp_read_worker(image_path_list, queue):
    data_home = "dataset/ImageNet22k"
    feature_list = []
    miss_feature_list = []
    feature_path_list = []

    for image_path in image_path_list:
        feature_path = "{}/{}".format(
            data_home,
            image_path.replace("images", "features").replace("JPEG", "npy"))
        if not os.path.exists(feature_path):
            miss_feature_list.append(feature_path)
            continue
        feature_path_list.append(feature_path)
        feature_i = np.load(feature_path).reshape((1, 1024))
        norm = np.sqrt(feature_i @feature_i.T)
        feature_i = feature_i / norm
        feature_list.append(feature_i)
    print('miss features: ', miss_feature_list)
    feature = np.concatenate(feature_list, axis=0)
    queue.put(feature)


def mp_read(image_path_list, num_workers=64):
    process_list = []
    queue_list = []
    for i in range(num_workers):
        queue = Queue()
        queue_list.append(queue)
        l = len(image_path_list)
        image_path_list_i = image_path_list[l * i // num_workers:l * (i + 1) //
                                            num_workers]
        p = Process(target=mp_read_worker, args=(image_path_list_i, queue))
        p.start()
        process_list.append(p)
    feature_list = []
    for i in range(num_workers):
        feature_list.append(queue_list[i].get())
        process_list[i].join()
    feature = np.concatenate(feature_list, axis=0)
    return feature


def main():
    list_file = "dataset/ImageNet22k/image_list_21k.txt"
    with open(list_file, 'r') as f:
        raw = f.read()
    image_path_list = [i.split(" ")[0] for i in raw.split('\n')]
    feature = mp_read(image_path_list)
    cluster(feature, image_path_list, 0, 3, len(image_path_list), 64)


if __name__ == '__main__':
    CLUSTER_ID = 0
    main()
