import os.path as osp
import os
import logging


class Duke(object):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        train_dir = os.path.join(self.dataset_dir,'bounding_box_train')
        test_dir = os.path.join(self.dataset_dir,'bounding_box_test')
        query_dir = os.path.join(self.dataset_dir,'query')
        train_heatmap_dir = os.path.join(self.dataset_dir, 'train-alphapose-result/heatmap')
        test_heatmap_dir = os.path.join(self.dataset_dir, 'test-alphapose-result/heatmap')
        query_heatmap_dir = os.path.join(self.dataset_dir, 'query-alphapose-result/heatmap')

        train_list, num_train_pids, num_train_imgs = self._process_data(train_dir,train_heatmap_dir,relabel=True)
        test_list, num_test_pids, num_test_imgs = self._process_data(test_dir,test_heatmap_dir)
        query_list, num_query_pids, num_query_imgs = self._process_data(query_dir,query_heatmap_dir)

        logging.info("=> DukeMTMS loaded")
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
    def _process_data(self,dir,hm_dir,relabel=False):
        train_list = []
        id_set = set()
        files = os.listdir(dir)
        for file in files:
            if not '.jpg' in file:
                continue
            img_path = os.path.join(dir,file)
            pid = int(file.split('_')[0])
            cid = int((file.split('c')[1]).split('_')[0])
            clothes_id = -1
            hm_path = os.path.join(hm_dir,os.path.splitext(file)[0]+'.npy')
            train_list.append((img_path,pid,cid,clothes_id,hm_path))
            id_set.add(pid)
        # relabel id to continues
        if relabel:
            id_list = list(id_set)
            id_list.sort()
            train_list_relabel = []
            for i in train_list:
                train_list_relabel.append((i[0],id_list.index(i[1]),i[2],i[3],i[4]))
            train_list = train_list_relabel
        return train_list, len(id_set), len(train_list)
