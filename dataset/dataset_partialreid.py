import os.path as osp
import os
import logging


class PartialReID(object):
    def __init__(self, dataset_dir):
        """
        The implementation of Occluded-ReID dataset loading
        download dataset with the guidance of https://github.com/lightas/ICCV19_Pose_Guided_Occluded_Person_ReID
        """
        self.dataset_dir = dataset_dir

        dir_gallery = os.path.join(dataset_dir, 'gallery')
        dir_query = os.path.join(dataset_dir, 'query')
        train_list, num_train_pids, num_train_imgs = 0, 751, 0
        test_list, num_test_pids, num_test_imgs = self._process_data(dir_gallery)
        query_list, num_query_pids, num_query_imgs = self._process_data(dir_query)

        logging.info("=> PartialReID loaded")
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
    def _process_data(self, dir_path):
        data_list_load = []
        id_set = set()
        for root, _, files in os.walk(dir_path):
            for file in files:
                if not '.jpg' in file:
                    continue
                img_path = os.path.join(root, file)
                cid = int(file.split('_')[1].split('c')[1])
                pid = int(file.split('_')[0])
                id_set.add(pid)
                data_list_load.append((img_path, pid, cid))
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
