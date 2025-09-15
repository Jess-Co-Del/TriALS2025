import torch
import os, json
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


class MySegmentation:
    def __init__(self):
        print('Model initialization done!')

    def process_image(self, input_image_file, output_path):

        venous, properties = SimpleITKIO().read_images([input_image_file])
        predictor = self.get_predictor()
        pred_np = predictor.predict_single_npy_array(venous, properties, None, None, False)
        print('Output shape = ', pred_np.shape)
        assert venous[0].shape[-3:] == pred_np.shape[-3:]
        return pred_np, properties

    def get_predictor(
        self,
        task="Dataset008_mixedlits",
        nnunet_model_dir='nnUNet_results',
        model_name='nnUNetTrainer__nnUNetConvnextPlans__3d_fullres',
        folds=(0,1,2,3,4)
    ):
        # network parameters
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device('cuda', 0),
            verbose=True,
            verbose_preprocessing=False,
            allow_tqdm=False
        )
        predictor.initialize_from_trained_model_folder(
            os.path.join(nnunet_model_dir,
                            f'{task}/{model_name}'),
            use_folds=folds,
            checkpoint_name='checkpoint_best.pth',
        )
        return predictor
