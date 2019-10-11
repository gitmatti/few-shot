from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
from skimage import io
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import pandas

import csv
from torchvision.datasets import VisionDataset
from functools import partial
import PIL.Image
from sklearn.preprocessing import LabelEncoder

from config import DATA_PATH


class OmniglotDataset(Dataset):
    def __init__(self, subset):
        """Dataset class representing Omniglot dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset

        self.df = pd.DataFrame(self.index_subset(self.subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

    def __getitem__(self, item):
        instance = io.imread(self.datasetid_to_filepath[item])
        # Reindex to channels first format as supported by pytorch
        instance = instance[np.newaxis, :, :]

        # Normalise to 0-1
        instance = (instance - instance.min()) / (instance.max() - instance.min())

        label = self.datasetid_to_class_id[item]

        return torch.from_numpy(instance), label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset):
        """Index a subset by looping through all of its files and recording relevant information.

        # Arguments
            subset: Name of the subset

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            Omniglot dataset dataset
        """
        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(DATA_PATH + '/Omniglot/images_{}/'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.png')])

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(DATA_PATH + '/Omniglot/images_{}/'.format(subset)):
            if len(files) == 0:
                continue

            alphabet = root.split('/')[-2]
            class_name = '{}.{}'.format(alphabet, root.split('/')[-1])

            for f in files:
                progress_bar.update(1)
                images.append({
                    'subset': subset,
                    'alphabet': alphabet,
                    'class_name': class_name,
                    'filepath': os.path.join(root, f)
                })

        progress_bar.close()

        return images


class MiniImageNet(Dataset):
    def __init__(self, subset):
        """Dataset class representing miniImageNet dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset

        self.df = pd.DataFrame(self.index_subset(self.subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Resize(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, item):
        instance = Image.open(self.datasetid_to_filepath[item])
        instance = self.transform(instance)
        label = self.datasetid_to_class_id[item]
        return instance, label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset):
        """Index a subset by looping through all of its files and recording relevant information.

        # Arguments
            subset: Name of the subset

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            miniImageNet dataset
        """
        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(DATA_PATH + '/miniImageNet/images_{}/'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.png')])

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(DATA_PATH + '/miniImageNet/images_{}/'.format(subset)):
            if len(files) == 0:
                continue

            class_name = root.split('/')[-1]

            for f in files:
                progress_bar.update(1)
                images.append({
                    'subset': subset,
                    'class_name': class_name,
                    'filepath': os.path.join(root, f)
                })

        progress_bar.close()
        return images


class DummyDataset(Dataset):
    def __init__(self, samples_per_class=10, n_classes=10, n_features=1):
        """Dummy dataset for debugging/testing purposes

        A sample from the DummyDataset has (n_features + 1) features. The first feature is the index of the sample
        in the data and the remaining features are the class index.

        # Arguments
            samples_per_class: Number of samples per class in the dataset
            n_classes: Number of distinct classes in the dataset
            n_features: Number of extra features each sample should have.
        """
        self.samples_per_class = samples_per_class
        self.n_classes = n_classes
        self.n_features = n_features

        # Create a dataframe to be consistent with other Datasets
        self.df = pd.DataFrame({
            'class_id': [i % self.n_classes for i in range(len(self))]
        })
        self.df = self.df.assign(id=self.df.index.values)

    def __len__(self):
        return self.samples_per_class * self.n_classes

    def __getitem__(self, item):
        class_id = item % self.n_classes
        return np.array([item] + [class_id]*self.n_features, dtype=np.float), float(class_id)


class FashionProductImages(VisionDataset):
    """TODO
    """
    base_folder = 'fashion-product-images-small'
    filename = "fashion-product-images-dataset.zip"
    url = None  # TODO
    file_list = None  # TODO

    top20_classes = [
        "Jeans", "Perfume and Body Mist", "Formal Shoes",
        "Socks", "Backpacks", "Belts", "Briefs",
        "Sandals", "Flip Flops", "Wallets", "Sunglasses",
        "Heels", "Handbags", "Tops", "Kurtas",
        "Sports Shoes", "Watches", "Casual Shoes", "Shirts",
        "Tshirts"]

    # Adapt syntax to few_shot github repo
    background_classes = [
        "Cufflinks", "Rompers", "Laptop Bag", "Sports Sandals", "Hair Colour",
        "Suspenders", "Trousers", "Kajal and Eyeliner", "Compact", "Concealer",
        "Jackets", "Mufflers", "Backpacks", "Sandals", "Shorts", "Waistcoat",
        "Watches", "Pendant", "Basketballs", "Bath Robe", "Boxers",
        "Deodorant", "Rain Jacket", "Necklace and Chains", "Ring",
        "Formal Shoes", "Nail Polish", "Baby Dolls", "Lip Liner", "Bangle",
        "Tshirts", "Flats", "Stockings", "Skirts", "Mobile Pouch", "Capris",
        "Dupatta", "Lip Gloss", "Patiala", "Handbags", "Leggings", "Ties",
        "Flip Flops", "Rucksacks", "Jeggings", "Nightdress", "Waist Pouch",
        "Tops", "Dresses", "Water Bottle", "Camisoles", "Heels", "Gloves",
        "Duffel Bag", "Swimwear", "Booties", "Kurtis", "Belts",
        "Accessory Gift Set", "Bra"
    ]

    evaluation_classes = [
        "Jeans", "Bracelet", "Eyeshadow", "Sweaters", "Sarees", "Earrings",
        "Casual Shoes", "Tracksuits", "Clutches", "Socks", "Innerwear Vests",
        "Night suits", "Salwar", "Stoles", "Face Moisturisers",
        "Perfume and Body Mist", "Lounge Shorts", "Scarves", "Briefs",
        "Jumpsuit", "Wallets", "Foundation and Primer", "Sports Shoes",
        "Highlighter and Blush", "Sunscreen", "Shoe Accessories",
        "Track Pants", "Fragrance Gift Set", "Shirts", "Sweatshirts",
        "Mask and Peel", "Jewellery Set", "Face Wash and Cleanser",
        "Messenger Bag", "Free Gifts", "Kurtas", "Mascara", "Lounge Pants",
        "Caps", "Lip Care", "Trunk", "Tunics", "Kurta Sets", "Sunglasses",
        "Lipstick", "Churidar", "Travel Accessory"
    ]

    # TODO.not_implemented: should different 'target_type' be allowed?
    target_type = 'articleType'

    def __init__(self, classes, root=DATA_PATH, split='all', transform=None,
                 target_transform=None, download=False):
        super(FashionProductImages, self).__init__(
            root, transform=transform, target_transform=target_transform)

        self.transform = transforms.Compose([
            transforms.Resize((80,60)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        assert split in ['train', 'test', 'all']
        self.split = split

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        fn = partial(os.path.join, self.root, self.base_folder)

        # TODO.refactor: refer to class attribute 'file_list' instead of "styles.csv"
        with open(fn("styles.csv")) as file:
            csv_reader = csv.reader(file)
            column_names = next(csv_reader)

        # additional column for comma artifacts in column 'productDisplayName'
        column_names.append(column_names[-1] + '2')

        # TODO.refactor: clean up column names, potentially merge last two columns
        self.df_meta = pandas.read_csv(fn("styles.csv"), names=column_names,
                                  skiprows=1)

        # relevant classes either by 'top'/'bottom' keyword or by list
        all_classes = set(self.df_meta[self.target_type])
        if classes is not None:
            if isinstance(classes, list):
                assert set(classes).issubset(all_classes)
            else:
                assert classes in ['top', 'bottom', 'background', 'evaluation']
                if classes == 'top':
                    classes = self.top20_classes
                elif classes == 'bottom':
                    classes = list(all_classes.difference(self.top20_classes))
                elif classes == 'background':
                    classes = self.background_classes
                else:
                    classes = self.evaluation_classes
        else:
            classes = list(all_classes)

        self.df_meta = self.df_meta.assign(
            filename=self.df_meta["id"].apply(lambda x: str(x) + ".jpg"))

        # parses out samples that
        # - have a the relevant class label
        # - have an image present in the 'images' folder
        # - confer to the given split 'train'/'test'
        images = os.listdir(fn("images"))
        if self.split == 'train':
            split_mask = self._train_mask(self.df_meta)
        elif self.split == 'test':
            split_mask = ~ self._train_mask(self.df_meta)
        else:
            split_mask = True

        self.df = self.df_meta.loc[
            (self.df_meta[self.target_type].isin(classes))
            & (self.df_meta["filename"].isin(images))
            & split_mask
            ]

        self.targets = self.df[self.target_type]
        self.target_codec = LabelEncoder()
        self.target_codec.fit(classes)

        self.target_indices = self.target_codec.transform(self.targets)
        self.n_classes = len(self.target_codec.classes_)

        # TODO extenstion to accomodate few_shot github repo
        # self.df = self.samples
        self.df = self.df.assign(my_id=self.df["id"])
        self.df = self.df.assign(id=np.arange(len(self.df)))
        self.df = self.df.assign(class_id=self.target_codec.transform(self.df[self.target_type]))

    def __getitem__(self, index):
        # TODO.check: some images are not RGB?
        # TODO.check: some images are not 80x60?
        sample = self.df["filename"].iloc[index]
        # sample = str(self.df["my_id"].iloc[index]) + ".jpg"
        X = PIL.Image.open(
            os.path.join(
                self.root,
                self.base_folder,
                "images",
                sample
            )
        ).convert("RGB")
        target = self.target_indices[index]

        # TODO.extension: allow returning one-hot representation of target

        if self.transform is not None:
            X = self.transform(X)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return X, target

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return self.n_classes

    def _train_mask(self, df):
        return df["year"] % 2 == 0

    def download(self):
        # TODO.not_implemented: check this and compare to e.g. MNIST/CIFAR
        # TODO.not_implemented: how to download from Kaggle
        # if self._check_integrity():
        #    print('Files already downloaded and verified')
        #    return

        # for (file_id, md5, filename) in self.file_list:
        #    download_file_from_google_drive(file_id, os.path.join(self.root, self.base_folder), filename, md5)

        # with zipfile.ZipFile(os.path.join(self.root, self.base_folder, "img_align_celeba.zip"), "r") as f:
        #    f.extractall(os.path.join(self.root, self.base_folder))

        raise NotImplementedError

    def _check_integrity(self):
        # TODO.not_implemented: check this and compare to e.g. MNIST/CIFAR

        # for (_, md5, filename) in self.file_list:
        #    fpath = os.path.join(self.root, self.base_folder, filename)
        #    _, ext = os.path.splitext(filename)
        #    # Allow original archive to be deleted (zip and 7z)
        #    # Only need the extracted images
        #    if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
        #        return False

        # Should check a hash of the images
        # return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))
        return True
