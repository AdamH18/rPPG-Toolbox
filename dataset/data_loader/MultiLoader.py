from torch.utils.data import Dataset
from dataset import data_loader
import itertools
import operator
import copy
import os

class MultiLoader(Dataset):
    def __init__(self, name, dataset_names, raw_data_paths, config_data, sec_pre, model):
        self.name = name
        self.loaders = list()
        if len(dataset_names) != len(raw_data_paths):
            raise ValueError("Different number of datasets and data paths")

        # Create each dataset loader based on passed-in list
        train_loaders = list()
        for ds in dataset_names:
            if ds == "UBFC-rPPG":
                train_loaders.append(data_loader.UBFCrPPGLoader.UBFCrPPGLoader)
            elif ds == "UBFC-rPPG1":
                train_loaders.append(data_loader.UBFCrPPG1Loader.UBFCrPPG1Loader)
            elif ds == "PURE":
                train_loaders.append(data_loader.PURELoader.PURELoader)
            elif ds == "SCAMPS":
                train_loaders.append(data_loader.SCAMPSLoader.SCAMPSLoader)
            elif ds == "MMPD":
                train_loaders.append(data_loader.MMPDLoader.MMPDLoader)
            elif ds == "BP4DPlus":
                train_loaders.append(data_loader.BP4DPlusLoader.BP4DPlusLoader)
            elif ds == "BP4DPlusBigSmall":
                train_loaders.append(data_loader.BP4DPlusBigSmallLoader.BP4DPlusBigSmallLoader)
            elif ds == "UBFC-PHYS":
                train_loaders.append(data_loader.UBFCPHYSLoader.UBFCPHYSLoader)
            elif ds == "COHFACE":
                train_loaders.append(data_loader.COHFACELoader.COHFACELoader)
            elif ds == "VICARPPG2":
                train_loaders.append(data_loader.VicarPPG2Loader.VicarPPG2Loader)
            else:
                raise ValueError("Unsupported dataset! Currently supporting UBFC-rPPG, PURE, MMPD, \
                                SCAMPS, BP4D+ (Normal and BigSmall preprocessing), and UBFC-PHYS.")

        print(f"Multiloader {self.name} beginning creation of data loaders:")
        for i in range(len(train_loaders)):
            print(f"Loader {dataset_names[i]} being created...")

            # Fix configs to save data in right place and save proper data file lists
            ds_config = copy.deepcopy(config_data)
            ds_config.defrost()
            path, file = os.path.split(config_data.CACHED_PATH)
            ds_config.CACHED_PATH = os.path.join(path, f"{dataset_names[i]}_{file.split('_')[1]}")
            fspath, fsfile = os.path.split(config_data.FILE_LIST_PATH)
            ds_config.FILE_LIST_PATH = os.path.join(fspath, f"{dataset_names[i]}_{'_'.join(fsfile.split('_')[1:])}")

            # Create loader for dataset with modified config
            self.loaders.append(train_loaders[i](name=f"train-{dataset_names[i]}",
                                data_path=raw_data_paths[i],
                                config_data=ds_config,
                                sec_pre=sec_pre,
                                model=model))
        
        # Calculate loader length in advance
        self.len = sum([len(l) for l in self.loaders])

        # Save order in which indexes should be accessed for each loader
        index_tups = list()
        for i, l in enumerate(self.loaders):
            index_tups.append([(i, j) for j in range(len(l))])
        self.indexes = list(MultiLoader.intersperse(*index_tups))
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        """Getter from each individual loader. Uses indexes list to spread datasets evenly"""
        return self.loaders[self.indexes[index][0]][self.indexes[index][1]]

    # https://stackoverflow.com/questions/19293481/how-to-elegantly-interleave-two-lists-of-uneven-length/59594546#59594546
    @staticmethod
    def distribute(sequence):
        """
        Enumerate the sequence evenly over the interval (0, 1).

        >>> list(distribute('abc'))
        [(0.25, 'a'), (0.5, 'b'), (0.75, 'c')]
        """
        m = len(sequence) + 1
        for i, x in enumerate(sequence, 1):
            yield i/m, x

    @staticmethod
    def intersperse(*sequences):
        """
        Evenly intersperse the sequences.

        Based on https://stackoverflow.com/a/19293603/4518341

        >>> list(intersperse(range(10), 'abc'))
        [0, 1, 'a', 2, 3, 4, 'b', 5, 6, 7, 'c', 8, 9]
        >>> list(intersperse('XY', range(10), 'abc'))
        [0, 1, 'a', 2, 'X', 3, 4, 'b', 5, 6, 'Y', 7, 'c', 8, 9]
        >>> ''.join(intersperse('hlwl', 'eood', 'l r!'))
        'hello world!'
        """
        distributions = map(MultiLoader.distribute, sequences)
        get0 = operator.itemgetter(0)
        for _, x in sorted(itertools.chain(*distributions), key=get0):
            yield x
