import os.path as osp
import os
from utils.logger import Logger
logging = Logger()


class OccludedReID(object):
    def __init__(self, dataset_dir):
        """
        The implementation of Occluded-ReID dataset loading
        """
        self.dataset_dir = dataset_dir
        gallery_dir = os.path.join(self.dataset_dir,'whole_body_images')
        query_dir = os.path.join(self.dataset_dir,'occluded_body_images')

        train_list, num_train_pids, num_train_imgs = 0, 751, 0
        test_list, num_test_pids, num_test_imgs = self._process_data(gallery_dir)
        query_list, num_query_pids, num_query_imgs = self._process_data(query_dir)

        logging.info("=> OccludedReID loaded")
        logging.info("Dataset statistics:")
        logging.info("  ------------------------------")
        logging.info("  subset   | # ids | # imgs")
        logging.info("  ------------------------------")
        logging.info("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        logging.info("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        logging.info("  gallery  | {:5d} | {:8d}".format(num_test_pids, num_test_imgs))
        logging.info("  ------------------------------")

        self.train = train_list
        self.num_train_pids = num_train_pids
        self.test = test_list
        self.query = query_list

    #img_paths, pid, cid, mask_paths
    def _process_data(self,dir,relabel=False):
        data_list_load = []
        id_set = set()
        files = os.listdir(dir)
        for root, dirs, files in os.walk(dir, topdown=False):
            for name in files:
                cid = -1
                img_path = os.path.join(root, name)
                pid = int(name.split('_')[0])
                id_set.add(pid)
                data_list_load.append((img_path, pid, cid))
        # relabel id to continues
        if relabel:
            id_list = list(id_set)
            id_list.sort()
            train_list_relabel = []
            for i in data_list_load:
                train_list_relabel.append((i[0],id_list.index(i[1]),i[2]))
            data_list_load = train_list_relabel
        # load as dict
        data_list = []
        for data in data_list_load:
            data_list.append(
                {
                    'img_path': data[0],
                    'pid': data[1],
                    'cid': data[2]
                }
            )
        return data_list, len(id_set), len(data_list)
