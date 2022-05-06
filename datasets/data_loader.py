from torch.utils.data.distributed import DistributedSampler
from ffrecord.torch import DataLoader
from hfai.datasets import LTSF


def get_dataloader(data_name: str, seq_len: int, label_len: int, pred_len: int, features: str, batch_size: int, num_workers: int=8, mode: str='train'):
    assert data_name in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2',
                         'exchange_rate', 'electricity',
                         'national_illness', 'traffic']
    assert mode in ['train', 'val']

    data = LTSF(
        data_name,
        split=mode,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        features=features,
    )
    datasampler = DistributedSampler(data, shuffle=True)
    dataloader = DataLoader(
        data, batch_size=batch_size, sampler=datasampler, num_workers=num_workers, pin_memory=True
    )

    x, y, x_mark, y_mark = data[[0]][0]
    encoder_dim = x.shape[-1]
    decoder_dim = y.shape[-1]
    output_dim = decoder_dim if features != 'MS' else 1

    return dataloader, data.get_scaler(), encoder_dim, decoder_dim, output_dim