from ray import tune
search_space = {
    "batch_size": tune.choice([16, 32]),
    "hidden_dim": tune.choice([256, 512, 1024]),
    "message_passing_steps": tune.choice([3, 4, 5, 6]),
    "edge_threshold_val": tune.uniform(3.0, 5.0),
    "message_net_dropout": tune.uniform(0.0, 0.3),
    "data_aug_factor": tune.uniform(1.0,5.0),  # + data augmentation!
    "message_net_layers": tune.choice([2, 3, 4]),

    "lr": tune.loguniform(1e-4, 1e-2),
    "weight_decay": tune.loguniform(1e-6, 1e-3),
    "num_epochs": 50,
    "grace_epochs": 5,

    "lr_factor": 0.5,
    "lr_patience": 3,
    "lr_threshold": 1e-3,
    "lr_cooldown": 2,
    "lr_min": 1e-6
}
