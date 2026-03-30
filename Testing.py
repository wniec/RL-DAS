from torch import nn

from agent import DQN, PPO
from env.cec_test_func import Schwefel
from env.ensemble import Ensemble
from trainer import Policy_train
from utils import WandbLogger
import time
from env.cec_dataset import Training_Dataset
import env
import os
import warnings
import torch
import numpy as np
import wandb


class Actor(nn.Module):
    def __init__(self, dim, optimizer_num, device):
        super().__init__()
        self.device = device
        self.embedders = nn.ModuleList([])
        for i in range(optimizer_num):
            self.embedders.append(
                (
                    nn.Sequential(
                        *[
                            nn.Linear(dim, 64),
                            nn.ReLU(),
                            nn.Linear(64, 1),
                            nn.ReLU(),
                        ]
                    )
                ).to(device)
            )
            self.embedders.append(
                nn.Sequential(
                    *[
                        nn.Linear(dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1),
                        nn.ReLU(),
                    ]
                ).to(device)
            )
        self.embedder_final = nn.Sequential(
            *[
                nn.Linear(9 + optimizer_num * 2, 64),
                nn.Tanh(),
            ]
        ).to(device)
        self.model = nn.Sequential(
            *[
                nn.Linear(64, 16),
                nn.Tanh(),
                nn.Linear(16, optimizer_num),
                nn.Softmax(),
            ]
        ).to(device)

    def forward(self, obs, test=False):
        feature = list(obs[:, 0])
        if not isinstance(feature, torch.Tensor):
            feature = torch.tensor(feature, dtype=torch.float).to(self.device)
        moves = []
        for i in range(len(self.embedders)):
            moves.append(
                self.embedders[i](
                    torch.tensor(list(obs[:, i + 1]), dtype=torch.float).to(self.device)
                )
            )
        moves = torch.cat(moves, dim=-1)
        batch = obs.shape[0]
        feature = torch.cat((feature, moves), dim=-1).view(batch, -1)
        feature = self.embedder_final(feature)
        logits = self.model(feature)
        if test:
            out = (feature.detach().cpu().tolist(), logits)
        else:
            out = logits
        return out


class PPO_critic(nn.Module):
    def __init__(self, dim, optimizer_num, device):
        super().__init__()
        self.device = device
        self.embedders = nn.ModuleList([])
        for i in range(optimizer_num):
            self.embedders.append(
                (
                    nn.Sequential(
                        *[
                            nn.Linear(dim, 64),
                            nn.ReLU(),
                            nn.Linear(64, 1),
                            nn.ReLU(),
                        ]
                    )
                ).to(device)
            )
            self.embedders.append(
                nn.Sequential(
                    *[
                        nn.Linear(dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1),
                        nn.ReLU(),
                    ]
                ).to(device)
            )
        self.embedder_final = nn.Sequential(
            *[
                nn.Linear(9 + optimizer_num * 2, 64),
                nn.Tanh(),
            ]
        ).to(device)
        self.model = nn.Sequential(
            *[
                nn.Linear(64, 16),
                nn.Tanh(),
                nn.Linear(16, 1),
            ]
        ).to(device)

    def forward(self, obs):
        feature = list(obs[:, 0])
        if not isinstance(feature, torch.Tensor):
            feature = torch.tensor(feature, dtype=torch.float).to(self.device)
        moves = []
        for i in range(len(self.embedders)):
            moves.append(
                self.embedders[i](
                    torch.tensor(list(obs[:, i + 1]), dtype=torch.float).to(self.device)
                )
            )
        moves = torch.cat(moves, dim=-1)
        batch = obs.shape[0]
        feature = torch.cat((feature, moves), dim=-1).view(batch, -1)
        feature = self.embedder_final(feature)
        batch = obs.shape[0]
        bl_val = self.model(feature.view(batch, -1))
        return bl_val


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    VectorEnv = env.SubprocVectorEnv
    problem = ["Schwefel"]
    subproblems = ["Ackley", "Ellipsoidal", "Griewank", "Rastrigin"]
    sublength = [0.1, 0.2, 0.2, 0.2, 0.3]
    Comp_lamda = [1, 10, 1]
    Comp_sigma = [10, 20, 30]
    indicated_dataset = None
    shifted = True
    rotated = True
    Train_set = 1024
    rl = "PPO"  # DQN / PPO
    buf_size = 30000
    rep_size = 3000
    dim = 10
    batch_size = 16
    MaxFEs = 200000
    period = 2500
    Epoch = 30
    epsilon = 0.3
    epsilon_decay = 0.99
    lr = 1e-5
    critic_lr = 1e-5
    lr_decay = 1
    sample_times = 2
    sample_size = -1
    sample_FEs_type = 2
    n_step = 5
    k_epoch = int(0.3 * (MaxFEs // period))
    save_internal = 5
    data_gen_seed = 2
    torch_seed = 1
    optimizers = ["NL_SHADE_RSP", "MadDE", "JDE21"]
    state_dict = None
    device = "cuda:0"
    run_time = time.strftime("%Y%m%dT%H%M%S")

    # TODO: Initialize wandb run here, e.g.:
    wandb.init(
        project="RL-DAS",
        name=run_time,
        config={
            "problem": problem,
            "dim": dim,
            "MaxFEs": MaxFEs,
            "period": period,
            "Epoch": Epoch,
            "lr": lr,
            "critic_lr": critic_lr,
            "k_epoch": k_epoch,
            "optimizers": optimizers,
            "rl": rl,
            "batch_size": batch_size,
        },
    )

    np.random.seed(data_gen_seed)
    torch.manual_seed(torch_seed)
    data_loader = Training_Dataset(
        filename=None,
        dim=dim,
        num_samples=Train_set,
        problems=problem,
        biased=False,
        shifted=shifted,
        rotated=rotated,
        batch_size=batch_size,
        save_generated_data=False,
        problem_list=subproblems,
        problem_length=sublength,
        indicated_specific=True,
        indicated_dataset=indicated_dataset,
    )
    ensemble = Ensemble(
        optimizers,
        Schwefel(dim, np.random.rand(dim), np.eye(dim), 0),
        period,
        MaxFEs,
        sample_times,
        sample_size,
    )
    np.random.seed(0)

    print("=" * 75)
    print("Running Setting:")
    print(
        ("Shifted " if shifted else "Unshifted ")
        + ("Rotated " if rotated else "Unrotated ")
        + f'Problem: {problem} with Dim: {dim}\n'
        f'Train Dataset: {Train_set}\n'
        f'MaxFEs: {MaxFEs} with Period: {period}\n'
        f'Feature Sample Times: {sample_times} with Sample Size: {sample_size if sample_size > 0 else "population"}\n'
        f'External FEs Type: {sample_FEs_type}\n'
        f'Optimizers: {optimizers}\n'
        f'Agent: {rl}\n'
        f'Replay Buffer: {buf_size}\n'
        f'Replay Size: {rep_size}\n'
        f'K Epoch: {k_epoch}\n'
        f'Batch Size: {batch_size}\n'
        f'Learning Rate: {lr} with decay: {lr_decay}\n'
        f'Epoch: {Epoch}\n'
        f'Device: {device}\n'
        f'Env: {VectorEnv.__name__}\n'
        f'Loaded Model: {state_dict}\n'
        f'Runtime: {run_time}'
    )
    print("=" * 75)

    state_shape = ensemble.observation_space.shape or ensemble.observation_space.n
    action_shape = ensemble.action_space.shape or ensemble.action_space.n

    # PPO
    net = Actor(dim, action_shape, device)
    critic = PPO_critic(dim, action_shape, device)
    if state_dict is not None:
        model = torch.load(state_dict, map_location=device)
        net.load_state_dict(model["actor"])
        critic.load_state_dict(model["critic"])
    optim = torch.optim.Adam(
        [{"params": net.parameters(), "lr": lr}]
        + [{"params": critic.parameters(), "lr": critic_lr}]
    )
    policy = PPO(net, critic, optim, device=device)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, lr_decay)

    logger = WandbLogger(train_interval=batch_size, update_interval=batch_size, project="RL-DAS")
    log_path = "save_policy_" + rl + "/" + run_time
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    policy.save(log_path, 0, run_time)

    avg_base_cost = 1e-8
    total_steps = 0
    train_steps = 0
    for epoch in range(Epoch):
        data_loader.shuffle()
        for bid, problems in enumerate(data_loader):
            envs = [
                lambda e=p: Ensemble(
                    optimizers,
                    e,
                    period,
                    MaxFEs,
                    sample_times,
                    sample_size,
                    sample_FEs_type=sample_FEs_type,
                    terminal_error=avg_base_cost,
                )
                for i, p in enumerate(problems)
            ]

            train_envs = VectorEnv(envs)
            batch_num = data_loader.N // data_loader.batch_size
            total_steps = Policy_train(
                policy,
                train_envs,
                logger,
                total_steps,
                train_steps,
                k_epoch,
                ensemble.max_step,
                epoch,
                bid,
                batch_num,
            )
            train_steps += k_epoch

            train_envs.close()

        epsilon *= epsilon_decay
        lr_scheduler.step()
        logger.write(
            "train/learning rate", epoch, {"train/lr": lr_scheduler.get_lr()[-1]}
        )
        wandb.log({"train/lr": lr_scheduler.get_lr()[-1]}, step=epoch)

        if (epoch + 1) % save_internal == 0:
            policy.save(log_path, epoch, run_time)
            # TODO: Log model checkpoint to wandb:
            # wandb.save(os.path.join(log_path, f"policy-{run_time}-{epoch}.pth"))
