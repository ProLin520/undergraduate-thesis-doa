import os, sys, json, argparse
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

ext_lib = r"D:\Python\Project\doa_estimation\Graduation\external\DOA_est_Master-master"
proj_root = r"D:\Python\Project\doa_estimation"
if ext_lib not in sys.path: sys.path.insert(0, ext_lib)
if proj_root not in sys.path: sys.path.insert(1, proj_root)
if "utils" in sys.modules: del sys.modules["utils"]

from data.data_create.signal_datasets90 import ULA_dataset, array_Dataloader
from data.data_create.Create_k_source_dataset90 import Create_random_k_input_theta, Create_datasets
from dl_models.vit_model import VisionTransformer
from dl_models.embeding_layer import scm_embeding, calc_rmse


VAL_SNR_LIST = [-20, -15, -10, -5, 0, 5, 10]
VAL_TRIPLETS = [(-8.5, 0.0, 8.5), (-10.5, 0.0, 10.5), (-12.5, 0.0, 12.5)]


def norm_scm(x):
    b = x.shape[0]
    m = torch.max(torch.abs(x.reshape(b, -1)), dim=1)[0].view(b, 1, 1, 1)
    return x / (m + 1e-8)


def valid_theta(theta, signal_range=(-90, 90)):
    theta = np.array(theta, dtype=np.float32)
    mask = (~np.isnan(theta).any(axis=1)) & (theta.max(axis=1) <= signal_range[1]) & (theta.min(axis=1) >= signal_range[0])
    return theta[mask]


def build_model(args):
    dim = 768
    return VisionTransformer(embed_layer=scm_embeding(args.M, dim), embed_dim=dim, out_dims=args.k, drop_ratio=0, attn_drop_ratio=0)


def base_cfg(epoch):
    if epoch < 30:
        return {"snr": (0, 10), "sep": 10}
    elif epoch < 70:
        return {"snr": (-10, 10), "sep": 7}
    else:
        return {"snr": (-20, 10), "sep": 5}


def make_random_loader(args, n, sep, snr, batch_size, out="doa"):
    dataset = ULA_dataset(args.M, args.signal_range[0], args.signal_range[1], args.step_used, args.rho)
    theta = Create_random_k_input_theta(args.k, args.signal_range[0], args.signal_range[1], n, min_delta_theta=sep)
    theta = valid_theta(theta, args.signal_range)
    Create_datasets(dataset, args.k, theta, batch_size=128, snap=args.snap, snr=snr, shared_snr=True)
    return array_Dataloader(dataset, batch_size, shuffle=True, load_style="torch", input_type="scm", output_type=out)


def make_val_loaders(args, batch_size=64, n=2000):
    theta = Create_random_k_input_theta(args.k, args.signal_range[0], args.signal_range[1], n, min_delta_theta=5)
    theta = valid_theta(theta, args.signal_range)

    loaders = {}
    for snr in VAL_SNR_LIST:
        dataset = ULA_dataset(args.M, args.signal_range[0], args.signal_range[1], args.step_used, args.rho)
        Create_datasets(dataset, args.k, theta.copy(), batch_size=128, snap=args.snap, snr=snr, shared_snr=True)
        loaders[snr] = list(array_Dataloader(dataset, batch_size, shuffle=False, load_style="torch", input_type="scm", output_type="doa"))
    return loaders


def sample_sym(n, center_range=(-5, 5), delta_range=(7, 13)):
    c = np.random.uniform(center_range[0], center_range[1], size=n)
    d = delta_range[0] + (delta_range[1] - delta_range[0]) * np.random.beta(1.7, 1.7, size=n)
    return np.stack([c - d, c, c + d], axis=1).astype(np.float32)


def make_ft_loader(args):
    dataset = ULA_dataset(args.M, args.signal_range[0], args.signal_range[1], args.step_used, args.rho)

    theta_rand = Create_random_k_input_theta(args.k, args.signal_range[0], args.signal_range[1], args.ft_rand_num, min_delta_theta=5)
    theta_rand = valid_theta(theta_rand, args.signal_range)
    theta_sym = sample_sym(args.ft_sym_num)

    theta = np.concatenate([theta_rand, theta_sym], axis=0)
    Create_datasets(dataset, args.k, theta, batch_size=128, snap=args.snap, snr=args.ft_snr, shared_snr=True)

    return array_Dataloader(dataset, args.ft_batch_size, shuffle=True, load_style="torch", input_type="scm", output_type="doa")


def make_sym_val_loaders(args):
    loaders = {}
    for triplet in VAL_TRIPLETS:
        dataset = ULA_dataset(args.M, args.signal_range[0], args.signal_range[1], args.step_used, args.rho)
        theta = np.tile(np.array([triplet], dtype=np.float32), (args.ft_val_num, 1))
        Create_datasets(dataset, args.k, theta, batch_size=128, snap=args.snap, snr=10.0, shared_snr=True)
        loaders[str(triplet)] = list(array_Dataloader(dataset, args.ft_val_batch_size, shuffle=False, load_style="torch", input_type="scm", output_type="doa"))
    return loaders


def sort_xy(pred, y):
    pred, _ = torch.sort(pred, dim=1)
    y, _ = torch.sort(y, dim=1)
    return pred, y


def sym_loss(pred, y):
    mask = torch.abs(y[:, 0] + y[:, 2] - 2 * y[:, 1]) < 1e-5
    if not torch.any(mask):
        return torch.tensor(0.0, device=pred.device)
    p = pred[mask]
    return torch.mean((p[:, 0] + p[:, 2] - 2 * p[:, 1]) ** 2)


def sym_success(pred, y):
    delta = y[:, 2] - y[:, 1]
    th = delta / 2
    err = torch.abs(pred - y)
    ok = (err[:, 0] < th) & (err[:, 1] < th) & (err[:, 2] < th)
    return ok.float().mean().item()


def train_epoch(model, loader, opt, loss_fn, device, epoch, ft=False, lam=0.08):
    model.train()
    total, steps = 0.0, 0
    pbar = tqdm(loader, file=sys.stdout)

    for x, y in pbar:
        x = norm_scm(x.to(device).float())
        y = y.to(device).float()

        pred = model(x)
        pred, y = sort_xy(pred, y)

        task = loss_fn(pred, y)
        loss = task + lam * sym_loss(pred, y) if ft else task

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1 if ft else 1.0)
        opt.step()

        total += task.item()
        steps += 1
        pbar.desc = f"[{'ft' if ft else 'base'} epoch {epoch}] loss: {task.item():.3f}"

    return total / steps


@torch.no_grad()
def eval_rand(model, loaders, device):
    model.eval()
    out = {}
    for snr, loader in loaders.items():
        mse, steps = 0.0, 0
        for x, y in loader:
            x = norm_scm(x.to(device).float())
            y = y.to(device).float()
            pred, y = sort_xy(model(x), y)
            mse += calc_rmse(pred, y)
            steps += 1
        out[snr] = float(np.sqrt(mse / steps))
    return float(np.mean(list(out.values()))), out


@torch.no_grad()
def eval_sym(model, loaders, device):
    model.eval()
    succ, rmse = {}, {}
    for name, loader in loaders.items():
        s, mse, steps = 0.0, 0.0, 0
        for x, y in loader:
            x = norm_scm(x.to(device).float())
            y = y.to(device).float()
            pred, y = sort_xy(model(x), y)
            s += sym_success(pred, y)
            mse += calc_rmse(pred, y)
            steps += 1
        succ[name] = s / steps
        rmse[name] = float(np.sqrt(mse / steps))
    return float(np.mean(list(succ.values()))), succ, float(np.mean(list(rmse.values())))


def freeze_for_ft(model):
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "blocks"):
        for p in model.blocks[-1].parameters():
            p.requires_grad = True
    if hasattr(model, "norm"):
        for p in model.norm.parameters():
            p.requires_grad = True
    if hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad = True


def main(args):
    os.makedirs(args.save_root, exist_ok=True)
    with open(os.path.join(args.save_root, "one_weight_train_set.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)

    save_path = os.path.join(args.save_root, "weight_base_ThreeSource.pth")
    loss_fn = torch.nn.MSELoss()

    model = build_model(args).to(args.device)

    print("\n========== Stage 1: Base ==========")
    opt = optim.Adam(model.parameters(), lr=args.base_lr)

    for epoch in range(args.base_epochs):
        cfg = base_cfg(epoch)
        loader = make_random_loader(args, args.base_num, cfg["sep"], cfg["snr"], args.base_batch_size)
        train_loss = train_epoch(model, loader, opt, loss_fn, args.device, epoch + 1, ft=False)
        print(f"[Base {epoch + 1}/{args.base_epochs}] TrainLoss: {train_loss:.4f} | snr={cfg['snr']} | sep={cfg['sep']}")

    print("\n========== Stage 2: Fine-tune ==========")
    freeze_for_ft(model)

    rand_val = make_val_loaders(args, batch_size=args.ft_val_batch_size, n=args.val_num)
    sym_val = make_sym_val_loaders(args)

    opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.ft_lr)
    best_score = -1e9

    for epoch in range(args.ft_epochs):
        loader = make_ft_loader(args)
        train_loss = train_epoch(model, loader, opt, loss_fn, args.device, epoch + 1, ft=True, lam=args.lambda_sym)

        rand_avg, snr_rmse = eval_rand(model, rand_val, args.device)
        sym_succ, triplet_succ, sym_rmse = eval_sym(model, sym_val, args.device)
        score = sym_succ - 0.005 * rand_avg

        print(f"[FT {epoch + 1}/{args.ft_epochs}] TrainLoss: {train_loss:.4f} | RandAvg: {rand_avg:.4f}° | SymSucc: {sym_succ:.4f} | SymRMSE: {sym_rmse:.4f}° | Score: {score:.4f}")

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), save_path)
            print(f"⭐ 保存最终模型: {save_path}")

    print(f"\n✅ 完成，只生成一个最终模型: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--snap", type=int, default=50)
    parser.add_argument("--signal_range", type=tuple, default=(-90, 90))
    parser.add_argument("--step_used", type=float, default=1)
    parser.add_argument("--rho", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_root", type=str, default=r"D:\Python\Project\doa_estimation\Graduation\result\vit\vit_M_8_k_3_base")

    parser.add_argument("--base_epochs", type=int, default=100)
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--base_num", type=int, default=10000)
    parser.add_argument("--base_batch_size", type=int, default=256)

    parser.add_argument("--ft_epochs", type=int, default=8)
    parser.add_argument("--ft_lr", type=float, default=5e-6)
    parser.add_argument("--ft_batch_size", type=int, default=64)
    parser.add_argument("--ft_rand_num", type=int, default=6000)
    parser.add_argument("--ft_sym_num", type=int, default=2500)
    parser.add_argument("--ft_snr", type=tuple, default=(-5.0, 10.0))
    parser.add_argument("--ft_val_num", type=int, default=1500)
    parser.add_argument("--ft_val_batch_size", type=int, default=64)
    parser.add_argument("--lambda_sym", type=float, default=0.08)
    parser.add_argument("--val_num", type=int, default=2000)

    args = parser.parse_args()
    main(args)