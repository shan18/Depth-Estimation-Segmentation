# Dataset

The dataset used for this model was taken from [here](https://www.kaggle.com/shanwizard/modest-museum-dataset). The dataset contains 400,000 images of backgrounds, background-foregrounds and their corresponding masks and depth maps each. For more info on the dataset, please go to this [link](https://github.com/shan18/MODEST-Museum-Dataset).

## Preview

The dataset contains four types of images

### Background

![bg](../images/dataset/bg_sample.png)

### Background-Foreground

![bg_fg](../images/dataset/bg_fg_sample.png)

### Background-Foreground Mask

![bg_fg_mask](../images/dataset/bg_fg_mask_sample.png)

### Background-Foreground Depth Map

![bg_fg_depth_map](../images/dataset/bg_fg_depth_map_sample.png)

## Preprocessing

- The input images (background and background-foreground) were normalized according to the values given on the dataset page.
- No preprocessing was done on the output images except converting them into torch.Tensor type and keeping their values within the range [0, 1].
- There was no point in applying any physical data augmentation techniques on the images as it would distort them from their corresponding labels which were not augmented.
- So, the only option left was to use photometric augmentations. I tried `HueSaturationValue` and `RandomContrast` augmentations from the `albumentations` package. The code for augmentation can be seen [here](../tensornet/data/processing.py).

## Data Loading

- The dataset is huge, so it is not possible to load the entire dataset into memory at once. So only the images names are indexed and they are fetched on need basis. The code for data loading can be found [here](../tensornet/data/datasets/modest_museum.py)
