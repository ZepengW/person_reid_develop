import os.path as osp
import os
import logging


class Market(object):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        train_dir = os.path.join(self.dataset_dir,'bounding_box_train')
        test_dir = os.path.join(self.dataset_dir,'bounding_box_test')
        query_dir = os.path.join(self.dataset_dir,'query')

        self.train, num_train_pids, num_train_imgs = self._process_data(train_dir,relabel=True)
        self.test, num_test_pids, num_test_imgs = self._process_data(test_dir)
        self.query, num_query_pids, num_query_imgs = self._process_data(query_dir)

        logging.info("=> Market1501 loaded")
        logging.info("Dataset statistics:")
        logging.info("  ------------------------------")
        logging.info("  subset   | # ids | # imgs")
        logging.info("  ------------------------------")
        logging.info("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        logging.info("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        logging.info("  gallery  | {:5d} | {:8d}".format(num_test_pids, num_test_imgs))
        logging.info("  ------------------------------")

        self.num_train_pids = num_train_pids


    def _process_data(self,dir,relabel=False):
        train_list = []
        id_set = set()
        files = os.listdir(dir)
        for file in files:
            if not '.jpg' in file:
                continue
            img_path = os.path.join(dir,file)
            pid = int(file.split('_')[0])
            if pid == -1:   # reduce the juck 
                continue
            cid = int((file.split('c')[1]).split('s')[0]) - 1
            train_list.append((img_path,pid,cid))
            id_set.add(pid)
        # relabel id to continues
        if relabel:
            id_list = list(id_set)
            id_list.sort()
            train_list_relabel = []
            for i in train_list:
                train_list_relabel.append((i[0],id_list.index(i[1]),i[2]))
            train_list = train_list_relabel
        # load as dict
        data_list = []
        for data in train_list:
            data_list.append(
                {
                    'img_path': data[0],
                    'pid': data[1],
                    'cid': data[2]
                }
            )
        return data_list, len(id_set), len(train_list)
