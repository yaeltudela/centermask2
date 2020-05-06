import os

from detectron2.evaluation import DatasetEvaluator


class GianaEvaulator(DatasetEvaluator):
    def __init__(self, dataset_name, output_dir, thresholds=None, old_metric=False):
        self.dataset_name = dataset_name
        self.output_folder = os.path.join(output_dir, "giana")
        self.detection_folder = os.path.join(output_dir, "detection")
        self.localization_folder = os.path.join(output_dir, "localization")
        self.classification_folder = os.path.join(output_dir, "classification")
        self.old_metric = old_metric



    def make_dirs(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        if not os.path.exists(self.detection_folder):
            os.makedirs(self.detection_folder)
        if not os.path.exists(self.localization_folder):
            os.makedirs(self.localization_folder)
        if not os.path.exists(self.classification_folder):
            os.makedirs(self.classification_folder)


    def reset(self):
        super().reset()

    def process(self, inputs, outputs):
        super().process(inputs, outputs)

    def evaluate(self):
        return super().evaluate()

