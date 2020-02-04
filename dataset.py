from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import glob
import os.path as osp
# from skimage import io
import lmdb
from io import BytesIO
import pyarrow as pa
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
expr_class = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']


class Aff2(Dataset):
    def __init__(self, transform=None, flag="train"):
        if flag == "train":
            db_path = r"/home/andy/newdata/train"
        else:
            db_path = r"/home/andy/newdata/val"
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path), max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pa.deserialize(txn.get(b'__len__'))
            self.keys = pa.deserialize(txn.get(b'__keys__'))

        # print(self.length)
        # print(type(self.keys))
        self.transform = transform
        # self.train_data = np.asarray(self.train_data)

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pa.deserialize(byteflow)
        # print("up:", type(unpacked))

        # load img
        imgbuf = unpacked[0]
        # TODO: can use BytesIO(imgbuf), better?
        buf = BytesIO(imgbuf)   # six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf, "r")

        # load label
        labels = unpacked[1]
        target = np.frombuffer(labels, dtype=np.float32)
        # print(target)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return self.length


def get_mean_std(dataset):
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=int(len(dataset)), shuffle=True, num_workers=8)
    print(len(dataloader))
    data = iter(dataloader).__next__()[0]
    print(data.shape)
    mean = np.mean(data.numpy(), axis=(0, 2, 3))
    std = np.std(data.numpy(), axis=(0, 2, 3))
    return mean, std


if __name__ == "__main__":
    import torchvision.transforms as transforms
    transform = transforms.ToTensor()
    mean, std = get_mean_std(Aff2(transform, "train"))
    print(mean, std)
#     F = Fer2D()
#     image, label = F.__getitem__(index=0)
#     print("img type", type(image))
#     print("label type", type(label))
#     print("label is ", label)
#     print("len", F.__len__())
#     print("len1", len(F.train_labels))
#     image.show()
