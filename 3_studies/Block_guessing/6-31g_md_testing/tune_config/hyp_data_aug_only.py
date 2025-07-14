from ray import tune
search_space = {
    "batch_size": 16,
    "hidden_dim": 256,
    "message_passing_steps": 4,
    "edge_threshold_val": 3,
    "message_net_dropout": 0.15,
    "data_aug_factor": tune.grid_search([1.0,1.5,2.0,2.5,3.0,3.5,4.0]),  # + data augmentation!
    "message_net_layers": 3,

    "lr": 2.68e-3,
    "weight_decay": 1.78e-5,
    "num_epochs": 50,
    "grace_epochs": 5,

    "lr_factor": 0.5,
    "lr_patience": 3,
    "lr_threshold": 1e-3,
    "lr_cooldown": 2,
    "lr_min": 1e-6
}
