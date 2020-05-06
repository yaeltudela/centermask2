import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json

dataset_annots = {
    'test-elipses': "test.json",
    'test-segmented': "test.json",
    'hisinvia': None,

    'CVC_ClinicDB': "clinic.json",
    'CVC_ColonDB': "colon.json",
    'CVC_HDClassif': "hdClassif.json",
    'CVC_VideoClinicDB_test': "test.json",
    'CVC_VideoClinicDB_train': "train.json",
    'CVC_VideoClinicDB_valid': "valid.json",
    'ETIS_LaribPolypDB': "etis.json",

}

polyp_categories = {
    "AD": {
        'id': 1,
        'name': 'AD',
        'supercategory': 'polyp',
    },
    "NAD": {
        'id': 2,
        'name': 'NAD',
        'supercategory': 'polyp',
    }
}


def register_coco_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, **metadata
    )


def register_polyp_datasets(dataset_list=None):
    DatasetCatalog.clear()
    if dataset_list is None:
        dataset_list = [os.path.join("datasets", x) for x in os.listdir("datasets") if
                        os.path.islink(os.path.join("datasets", x))]

    for dataset in dataset_list:
        dataset_name = str(dataset.split("/")[-1])
        json_file = dataset_annots[dataset_name]
        if json_file is None:
            continue
        else:
            json_file = os.path.join(dataset, "annotations", json_file)
        image_root = os.path.join(dataset, "images")

        thing_ids = [v["id"] for k, v in polyp_categories.items() if k in ["AD", "NAD"]]
        # Mapping from the incontiguous COCO category id to an id in [0, 79]
        thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
        thing_classes = [v["name"] for k, v in polyp_categories.items() if k in ["AD", "NAD"]]

        metadata = {
            "thing_classes": thing_classes,
            "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
            "evaluator_type": 'giana' if "test" in dataset_name else "coco",
        }
        register_coco_instances(dataset_name, metadata, json_file, image_root)
        print("Dataset {} registered".format(dataset_name))


if __name__ == '__main__':
    register_polyp_datasets()



