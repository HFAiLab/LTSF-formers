import hf_env
hf_env.set_env("202111")

import os
import time
import numpy as np
from pathlib import Path
import torch
from torch.nn.parallel import DistributedDataParallel

import hfai
import hfai.nccl.distributed as dist
from torch.multiprocessing import Process
hfai.client.bind_hf_except_hook(Process)

from datasets import get_dataloader
from models import Informer, InformerStack, Autoformer



def train_one_epoch(epoch, start_step, features, model, train_loader, optimizer, criterion, standard_scaler, save_path, best_mse):
    model.train()
    for step, batch in enumerate(train_loader):
        if step < start_step:
            continue

        optimizer.zero_grad()

        x, y, x_mark, y_mark = [x.float().cuda(non_blocking=True) for x in batch]

        # decoder input
        dec_inp = torch.zeros([y.shape[0], model.pred_len, y.shape[-1]]).float()
        dec_inp = torch.cat([y[:, : model.label_len, :], dec_inp], dim=1).float().cuda(non_blocking=True)
        # encoder - decoder
        outputs = model(x, x_mark, dec_inp, y_mark)
        y_pred = standard_scaler.inverse_transform(outputs)
        f_dim = -1 if features == "MS" else 0
        y_true = y[:, -model.pred_len:, f_dim:].float().cuda(non_blocking=True)

        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()

        # 收到打断信号，保存模型，设置当前执行的状态信息
        rank = dist.get_rank()
        if rank == 0 and hfai.receive_suspend_command():
            state = {
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "step": step,
                "best_mse": best_mse,
            }
            torch.save(state, save_path / "latest.tar")
            time.sleep(5)
            hfai.go_suspend()


def eval(model, criterion, features, eval_loader, standard_scaler):
    loss, total = torch.zeros(2).cuda()

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            x, y, x_mark, y_mark = [x.float().cuda(non_blocking=True) for x in batch]

            # decoder input
            dec_inp = torch.zeros([y.shape[0], model.pred_len, y.shape[-1]]).float()
            dec_inp = torch.cat([y[:, : model.label_len, :], dec_inp], dim=1).float().cuda(non_blocking=True)
            # encoder - decoder
            outputs = model(x, x_mark, dec_inp, y_mark)
            y_pred = standard_scaler.inverse_transform(outputs)
            f_dim = -1 if features == "MS" else 0
            y_true = y[:, -model.pred_len:, f_dim:].float().cuda(non_blocking=True)

            loss += criterion(y_pred, y_true)
            total += y_true.size(0)

    for x in [loss, total]:
        dist.reduce(x, 0)

    loss_val = 0
    if dist.get_rank() == 0:
        loss_val = loss.item() / total.item()
    return loss_val


def fit(model, optimizer, criterion, train_loader, val_loader, save_path, features='S', epochs=100, standard_scaler=None):
    best_mse = np.inf

    if standard_scaler is None:
        raise RuntimeError("The standard scaler is None.")

    rank = dist.get_rank()

    # 如果模型存在checkpoint
    start_epoch, start_step = 0, 0
    if Path(save_path / "latest.tar").exists():
        ckpt = torch.load(save_path / "latest.tar", map_location="cpu")
        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        start_step = ckpt["step"]
        best_mse = ckpt["best_mse"]

    # 训练、验证
    for epoch in range(start_epoch, epochs):
        train_loader.sampler.set_epoch(epoch)

        train_one_epoch(epoch, start_step, features, model, train_loader, optimizer, criterion, standard_scaler, save_path, best_mse)
        train_loss = eval(model, criterion, features, train_loader, standard_scaler)
        val_loss = eval(model, criterion, features, val_loader, standard_scaler)

        # 保存
        if rank == 0:
            print(f"Epoch: {epoch}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

            if val_loss < best_mse:
                best_mse = val_loss
                print(f"New Best MSE: {best_mse:.4f}!")
                torch.save(model.module.state_dict(), save_path / "best.pt")

        torch.cuda.empty_cache()


def main(local_rank):
    # 超参数设置
    epochs = 100
    batch_size = 64
    num_workers = 4
    lr = 0.0001
    data_name = 'ETTh1'
    seq_len = 96
    label_len = 48
    pred_len = 48
    features = 'S'

    save_path = Path(f"output/{data_name}")
    save_path.mkdir(exist_ok=True, parents=True)

    # 多机通信
    ip = os.environ.get("MASTER_ADDR", '127.0.0.1')
    port = os.environ.get("MASTER_PORT" '8899')
    hosts = int(os.environ.get("WORLD_SIZE", '1'))  # 机器个数
    rank = int(os.environ.get("RANK", '0'))  # 当前机器编号
    gpus = torch.cuda.device_count()  # 每台机器的GPU个数

    # world_size是全局GPU个数，rank是当前GPU全局编号
    dist.init_process_group(
        backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts * gpus, rank=rank * gpus + local_rank
    )
    torch.cuda.set_device(local_rank)

    train_loader, standard_scaler, encoder_dim, decoder_dim, output_dim = get_dataloader(data_name, seq_len, label_len, pred_len, features, batch_size, num_workers, mode='train')
    val_loader, _, _, _, _ = get_dataloader(data_name, seq_len, label_len, pred_len, features, batch_size, num_workers, mode='val')

    model = Autoformer(
        enc_in=encoder_dim,
        dec_in=decoder_dim,
        c_out=output_dim,
        seq_len=seq_len,
        label_len=label_len,
        out_len=pred_len,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        factor=3,
        dropout=0.05,
        embed="timeF"
    )

    model = DistributedDataParallel(model.cuda(), device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    fit(model, optimizer, criterion, train_loader, val_loader, epochs, standard_scaler)


if __name__ == "__main__":
    ngpus = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(), nprocs=ngpus)
