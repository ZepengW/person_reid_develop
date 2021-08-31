import os.path as osp
import os
import logging


'''
This is a Clothes Changing Dataset Published in ACCV 2020
Evaluation Settings:
    use both cloth-consistent and cloth-changing samples in train set for training
    for testing, including:
    -Standard Setting:
        like standard dataset evaluation, the images in the test set with the same identity and the same camera view
        are discarded when computing evaluation scores
    -Cloth-changing Setting:
        the images with same identity, camera view and clothes are discarded during testing
'''
class LTCC(object):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        train_dir = os.path.join(self.dataset_dir,'train')
        test_dir = os.path.join(self.dataset_dir,'test')
        query_dir = os.path.join(self.dataset_dir,'query')

        self.train, num_train_pids, num_train_imgs = self._process_data(train_dir,relabel=True)
        self.test, num_test_pids, num_test_imgs = self._process_data(test_dir)
        self.query, num_query_pids, num_query_imgs = self._process_data(query_dir)

        logging.info("=> LTCC ReID loaded")
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
        data_list = []
        id_set = set()
        files = os.listdir(dir)
        for file in files:
            if not '.png' in file:
                continue
            img_path = os.path.join(dir,file)
            p_id = int(file.split('_')[0])
            clothes_id = int(file.split('_')[1])
            camera_id = int(file.split('_')[2].split('c')[1]) - 1
            mask_path = os.path.join(dir+'-mask',file)
            mask_contour_path = os.path.join(dir + '_rend', file)
            data_list.append((img_path,p_id,camera_id,clothes_id,(mask_path,mask_contour_path)))
            id_set.add(p_id)
        # relabel id to continues
        if relabel:
            id_list = list(id_set)
            id_list.sort()
            train_list_relabel = []
            for i in data_list:
                train_list_relabel.append((i[0],id_list.index(i[1]),i[2],i[3],i[4]))
            data_list = train_list_relabel
        return data_list, len(id_set), len(data_list)
