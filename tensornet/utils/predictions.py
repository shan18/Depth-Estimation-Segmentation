import torch


def class_level_accuracy(model, loader, device, classes):
    """Print test accuracy for each class in dataset.

    Args:
        model (torch.nn.Module): Model Instance.
        loader (torch.utils.data.DataLoader): Data Loader
        device (str or torch.device): Device where data will be loaded.
        classes (list or tuple): List of classes in the dataset
    """

    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))

    with torch.no_grad():
        for _, (images, labels) in enumerate(loader, 0):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()

            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # Display class level accuracy
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


def get_predictions(model, loader, device, sample_count=25):
    """Get correct and incorrect model predictions.

    Args:
        model (torch.nn.Module): Model Instance.
        loader (torch.utils.data.DataLoader): Data Loader.
        device (str or torch.device): Device where data will be loaded.
        sample_count (int, optional): Total number of predictions to store from
            each correct and incorrect samples. (default: 25)
    """

    correct_samples = []
    incorrect_samples = []

    with torch.no_grad():
        for _, (images, labels) in enumerate(loader, 0):
            img_batch = images  # This is done to keep data in CPU
            images, labels = images.to(device), labels.to(device)  # Get samples
            output = model(images)  # Get trained model output
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            result = pred.eq(labels.view_as(pred))

            # Save correct and incorrect samples
            correct_complete = False
            incorrect_complete = False
            for i in range(len(list(result))):
                if list(result)[i]:
                    if len(correct_samples) < sample_count:
                        correct_samples.append({
                            'id': i,
                            'image': img_batch[i],
                            'prediction': list(pred)[i],
                            'label': list(labels.view_as(pred))[i],
                        })
                    else:
                        correct_complete = True
                else:
                    if len(incorrect_samples) < sample_count:
                        incorrect_samples.append({
                            'id': i,
                            'image': img_batch[i],
                            'prediction': list(pred)[i],
                            'label': list(labels.view_as(pred))[i],
                        })
                    else:
                        incorrect_complete = True
            
            if correct_complete and incorrect_complete:
                break
    
    return correct_samples, incorrect_samples
