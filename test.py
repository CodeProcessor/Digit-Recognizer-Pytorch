"""
@Filename:    test.py.py
@Author:      dulanj
@Time:        2021-08-23 16.12
"""
import datetime

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import DigitRecognizerDataset
from model import ClassificationModel
from params import NUM_WORKERS, PIN_MEMORY, TEST_CSV, LOAD_MODEL_FILE, NO_OF_CLASSES, DEVICE
from train import transform


def test():
    model = ClassificationModel(num_classes=NO_OF_CLASSES).to(DEVICE)
    checkpoint = torch.load(LOAD_MODEL_FILE)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_dataset = DigitRecognizerDataset(
        transform=transform,
        file_path=TEST_CSV,
        test=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False
    )

    df = pd.DataFrame(columns=['ImageId', 'Label'])
    _df_index = 0
    for x, _id in test_loader:
        x = x.to(DEVICE)
        outputs = model(x)

        _, pred = torch.max(outputs.data, 1)


        prediction = pred.detach().cpu().numpy()[0]
        index = _id.detach().cpu().numpy()[0]
        df.loc[_df_index] = [index, prediction]
        _df_index += 1

        print(_df_index)
        # print(f"{index} - {prediction}")

        # tensor_image = torch.squeeze(x).to('cpu')
        # plt.imshow(tensor_image)
        # plt.show()

    now = datetime.datetime.now()
    date_time_str = now.strftime("%Y%m%d-%H%M%S")
    df.to_csv(f"submission_{date_time_str}.csv", index=False)


if __name__ == '__main__':
    test()
