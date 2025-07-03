from ray import tune
search_space = {
    "batch_size": tune.choice([2, 4, 8, 16, 32]),
    "hidden_dim": tune.choice([128, 256, 512]),
    "message_passing_steps": tune.choice([1, 2, 3]),
    "edge_threshold_val": tune.uniform(2.0, 4.0),
    "message_net_layers": tune.choice([1, 2, 3]),
    "message_net_dropout": tune.uniform(0.0, 0.3),
    "data_aug_factor": tune.uniform(1.0, 3.0),

    "lr": tune.loguniform(1e-4, 1e-2),
    "weight_decay": tune.loguniform(1e-6, 1e-3),
    "num_epochs": 30,
    "grace_epochs": 5,

    "lr_factor": 0.5,
    "lr_patience": 3,
    "lr_threshold": 1e-3,
    "lr_cooldown": 2,
    "lr_min": 1e-6
}
