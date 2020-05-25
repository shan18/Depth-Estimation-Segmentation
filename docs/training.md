# Training

After finalizing the model architecture, dataset and loss functions, we can now start training the model. In [this](deciding_loss_function.md#depth-and-segmentation-prediction) section, we selected two loss functions: **RMSE + (BCE & Dice)** and **SSIM + Dice**. So, we'll train our model with those two loss functions.

Each experiment described below had the following common traits

- The model was trained on smaller resolution images first and then gradually the image resolution was increased.
- **IoU** and **RMSE** were used as evaluation metrics. IoU was calculated on mask outputs while rmse was calculated on depth outputs.
- TensorBoard integration which generated model output after every epoch to keep track of model progress.
- Reduce LR on Plateau with patience of 2 and min lr of 1e-6.
- Auto model checkpointing which saved the model weights after every epoch.
- Each model was trained in small sets of epochs, this was done to ensure that the model training does not suffer from sudden disconnection from Google Colab.

## RMSE + (BCE & Dice)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c_ZnY4V3MJwe8re6PoYlrqnMCSEsMWFI?usp=sharing)

This experiment had the following parameters and hyperparameters:

- Data Augmentation
  - Hue Saturation Value
  - Random Contrast
- Image resolution change during training
  - First 3 epochs: 96x96
  - Epoch 4-8: 160x160
  - Epoch 9-11: 224x224

The code for the experiment can be found in the Google Colab link mentioned above as well as [here](../trial_notebooks/DES_RMSE_BCE_Dice.ipynb).

### Results

|                       RMSE                       |                      IoU                       |
| :----------------------------------------------: | :--------------------------------------------: |
| ![rmse](../images/rmse_bce_dice/rmse_change.png) | ![iou](../images/rmse_bce_dice/iou_change.png) |

### Predictions

## SSIM + Dice

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QmTXVlEM4zjIQ4_sxjsTVVVzPkkdErPs?usp=sharing)

This experiment had the following parameters and hyperparameters:

- Data Augmentation
  - Hue Saturation Value
  - Random Contrast
- Image resolution change during training
  - First 3 epochs: 96x96
  - Epoch 4-8: 160x160

The code for the experiment can be found in the Google Colab link mentioned above as well as [here](../trial_notebooks/DES_SSIM_DICE.ipynb).

### Results

|                     RMSE                     |                    IoU                     |
| :------------------------------------------: | :----------------------------------------: |
| ![rmse](../images/ssim_dice/rmse_change.png) | ![iou](../images/ssim_dice/iou_change.png) |

### Predictions
