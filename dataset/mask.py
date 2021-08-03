import os
from PIL import Image
import numpy as np

MASK_MAP = [
    # body
    ['Hair','Face', 'Left-arm', 'Right-arm', 'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe'],
    # clothes
    ['Hat', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat', 'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt'],
    #background
    ['Background']
]
LIP_LABEL = ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']

def read_person_mask(mask_path):
    if not os.path.exists(mask_path):
        return None
    with Image.open(mask_path) as mask_img:
        mask_array = np.array(mask_img,dtype=np.uint8)
    mask_out = np.zeros([3,mask_array.shape[0],mask_array.shape[1]],dtype=np.bool)
    for i in range(0,3):
        for label in MASK_MAP[i]:
            s = mask_array == LIP_LABEL.index(label)
            mask_out[i] = np.bitwise_or(mask_out[i],s)
    mask_out = mask_out.astype(np.uint8)
    return mask_out


if __name__ == '__main__':
    mask = read_person_mask('../../DATASET/LTCC_ReID/train-mask/000_1_c2_001038.png')
    output_img = Image.fromarray(np.asarray(mask[0], dtype=np.uint8))
    output_img.putpalette([0,0,0,255,0,0])
    output_img.save('./body.png')
    output_img = Image.fromarray(np.asarray(mask[1], dtype=np.uint8))
    output_img.putpalette([0, 0, 0, 255, 0, 0])
    output_img.save('./clothes.png')
    output_img = Image.fromarray(np.asarray(mask[2], dtype=np.uint8))
    output_img.putpalette([0, 0, 0, 255, 0, 0])
    output_img.save('./background.png')
    print('finish')