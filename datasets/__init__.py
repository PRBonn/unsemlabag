from datasets.MapData import MapData
from datasets.PhenoBench import PhenoBench


def get_dataset(name, dataset_opts):
    if name == "pb":
        return PhenoBench(dataset_opts)
    elif name == "map":
        return MapData(dataset_opts)
    else:
        raise RuntimeError("Dataset {} not available".format(name))
