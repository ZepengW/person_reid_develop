'''
reading dataset occ reid
according to https://github.com/wangguanan/light-reid/tree/master/examples/occluded_reid
using market1501 as training set
'''
import os
import logging

class OccludedReID(object):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        train_dir = os.path.join(self.dataset_dir, 'market1501/bounding_box_train')
        test_dir = os.path.join(self.dataset_dir, 'whole_body_images')
        query_dir = os.path.join(self.dataset_dir, 'occluded_body_images')

        self.train, num_train_pids, num_train_imgs = self._process_data_train(train_dir)
        self.test, num_test_pids, num_test_imgs = self._process_data(test_dir)
        self.query, num_query_pids, num_query_imgs = self._process_data(query_dir)

        logging.info("=> OccludedReID(test) and Market1501(train) loaded")
        logging.info("Dataset statistics:")
        logging.info("  ------------------------------")
        logging.info("  subset            | # ids | # imgs")
        logging.info("  ------------------------------")
        logging.info("  train(market1501) | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        logging.info("  query             | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        logging.info("  gallery           | {:5d} | {:8d}".format(num_test_pids, num_test_imgs))
        logging.info("  ------------------------------")
        self.num_train_pids = num_train_pids

    def _process_data_train(self, dir):
        train_list = []
        id_set = set()
        files = os.listdir(dir)
        for file in files:
            if not '.jpg' in file:
                continue
            img_path = os.path.join(dir, file)
            pid = int(file.split('_')[0])
            cid = int((file.split('c')[1]).split('s')[0]) - 1
            train_list.append((img_path, pid, cid))
            id_set.add(pid)
        # relabel id to continues
        id_list = list(id_set)
        id_list.sort()
        train_list_relabel = []
        for i in train_list:
            train_list_relabel.append(
                {
                    'img_path': i[0],
                    'pid': id_list.index(i[1]),
                    'cid': -1
                }
            )
        train_list = train_list_relabel
        return train_list, len(id_set), len(train_list)

    def _process_data(self, dir):
        data_list = []
        id_set = set()
        dirs_pid = os.listdir(dir)
        for dir_pid in dirs_pid:
            dir_path = os.path.join(dir, dir_pid)
            if not os.path.isdir(dir_path):
                continue
            pid = int(dir_pid)
            id_set.add(pid)
            for file_name in os.listdir(dir_path):
                if not '.tif' in file_name:
                    continue
                img_path = os.path.join(dir_path, file_name)
                data_list.append(
                    {
                        'img_path': img_path,
                        'pid': pid,
                        'cid': -1
                    }
                )
        return data_list, len(id_set), len(data_list)