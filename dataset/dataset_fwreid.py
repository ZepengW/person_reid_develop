import os.path as osp
import os
from utils.logger import Logger
logging = Logger()


class FWreID(object):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        train_dir = os.path.join(self.dataset_dir,'bounding_box_train')
        test_dir = os.path.join(self.dataset_dir,'bounding_box_test')
        query_dir = os.path.join(self.dataset_dir,'query')

        self.train, num_train_pids, num_train_imgs = self._process_traindir(train_dir,relabel=True)
        self.test, num_test_pids, num_test_imgs = self._process_testdir(test_dir)
        self.query, num_query_pids, num_query_imgs = self._process_testdir(query_dir)

        logging.info("=> FRreID loaded")
        logging.info("Dataset statistics:")
        logging.info("  ------------------------------")
        logging.info("  subset   | # ids | # imgs")
        logging.info("  ------------------------------")
        logging.info("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        logging.info("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        logging.info("  gallery  | {:5d} | {:8d}".format(num_test_pids, num_test_imgs))
        logging.info("  ------------------------------")

        self.num_train_pids = num_train_pids

    def _process_traindir(self, dir_path, relabel=False):
        imgs = os.listdir(dir_path)
        pid_container = set()
        dataset = []
        for img in imgs:
            pid = int(img[0:4]) - 95
            pid_container.add(pid)
            dataset.append(
                {
                    'img_path': dir_path + '/' + img,
                    'pid': pid,
                    'cid': 0
                }
            )
        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs

    def _process_testdir(self, dir_path, relabel=False):
        imgs = os.listdir(dir_path)
        pid_container = set()
        dataset = []
        for img in imgs:
            pid = int(img[0:4])
            pid_container.add(pid)
            dataset.append(
                {
                    'img_path': dir_path + '/' + img,
                    'pid': pid,
                    'cid': 0
                }
            )

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs
