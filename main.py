from datetime import datetime
from tqdm import tqdm

from settings import configs
from utils import dataset
from utils.dataset import prepare_data_loaders, get_data_loaders

def main():

    prepare_data_loaders(configs)

    for epoch in range(10):

        train_data_loader, valid_data_loader = get_data_loaders(configs)

        t0 = datetime.now()

        for batch in tqdm(train_data_loader):
            # batch = (path_list, mel_batch, encoded_batch, text_list, mel_length_list, encoded_length_list)
            # Check collate_function in utils/dataset.py
            path_list, mel_batch, encoded_batch, text_list, mel_length_list, encoded_length_list = batch
            
            pass

        t1 = datetime.now()

        for batch in tqdm(valid_data_loader):

            pass

        t2 = datetime.now()

        print(t1 - t0, t2 - t1)


    return

if __name__ == "__main__":
    
    main()