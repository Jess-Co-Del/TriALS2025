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

    def process_image(self, input_image_file, input_click_path, output_path, n_clicks):
        tumor_clicks, bg_clicks = self.get_coords(input_click_path, n_clicks)
        self.save_heatmap(
            input_image_file,
            tumor_clicks,
            bg_clicks,
            input_image_file.replace("_0000.nii.gz", "_0001.nii.gz")
        )

        venous, properties = SimpleITKIO().read_images([
            input_image_file, input_image_file.replace("_0000.nii.gz", "_0001.nii.gz")])
        images = np.stack([venous[0], venous[1]])
        predictor = self.get_predictor()
        pred_np = predictor.predict_single_npy_array(images, properties, None, None, False)
        # pred = SimpleITKIO().read_images([
        #     os.path.join(output_path, os.path.basename(input_image_file).replace("_0000.nii.gz", ".nii.gz"))])
        print('Output shape = ', pred_np.shape)
        assert venous[0].shape[-3:] == pred_np.shape[-3:]
        return pred_np, properties

    def get_coords(self, json_path, n_clicks: int = 10):
        json_file = json.load(open(json_path, "rb"))
        try:
            tumor_coords = json_file['lesion'][:n_clicks]
        except:
            tumor_coords = json_file['tumor'][:n_clicks]
        bg_coords = json_file['background'][:n_clicks]

        return tumor_coords, bg_coords

    def save_heatmap(self, ct_path, tumor_clicks, bg_clicks, output_path):
        ct_nii = nib.load(ct_path)
        ref = ct_nii.get_fdata()
        ref_shape = ref.shape
        ref_affine = ct_nii.affine

        if len(tumor_clicks) == 0:
            heatmap = np.zeros(ref_shape, dtype=np.float32)
        else:
            point_map_neg = np.zeros(ref_shape, dtype=np.float32)
            heatmap_pos = np.zeros(ref_shape, dtype=np.float32)
            for t_click in tumor_clicks:
                point_map_pos = np.zeros(ref_shape, dtype=np.float32)
                point_map_pos[(t_click[0],t_click[1],t_click[2])]=1.
                suv_val_at_click = ref[t_click[0],t_click[1],t_click[2]]
                local_gauss = gaussian_filter(point_map_pos, 3)    
                # Set to zero points where SUV is lower than 4 or than SUV click
                local_gauss[ref<min(4, suv_val_at_click)] = 0
                max_pos = np.max(local_gauss)
                local_gauss/=max_pos
                heatmap_pos +=local_gauss
                heatmap_pos[heatmap_pos>1]=1
            for b_click in bg_clicks:
                point_map_neg[(b_click[0],b_click[1],b_click[2])]=1.

            heatmap_neg =  gaussian_filter(point_map_neg, 6)

            # Set to zeros points that are in the positive gaussian map but not a background click
            heatmap_neg[np.logical_and(heatmap_pos!=0, point_map_neg==0)] = 0  

            # Normalizing between 0 and 1
            max_pos = np.max(heatmap_pos)
            max_neg = np.max(heatmap_neg)

            if max_pos and max_neg:
                heatmap = heatmap_pos/max_pos - heatmap_neg/max_neg
            elif max_pos:  # There are no background points
                heatmap = heatmap_pos/max_pos
            elif max_neg:  # There are no foreground points
                heatmap = heatmap_neg/max_neg
            else:
                heatmap = heatmap_pos/max_pos

        out = nib.Nifti1Image(heatmap, affine=ref_affine)
        nib.save(out, output_path)

        return heatmap

    def get_predictor(
        self,
        task="Dataset005_trialsclick",
        nnunet_model_dir='nnUNet_results',
        model_name='nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres',
        folds=(1, 2, 3, 4)
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
