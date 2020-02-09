max_environment_steps = 2048
num_actions = 4
num_observations = 24
weight_std = 0.001
max_task_time_sec = 120
model_dir = './models'
model_namefile = './model_namefile.txt'
bucket_name = 'danielsmith-poet-models'
layer_definition = [
    64,
    64
]
env_name = 'BipedalWalker-v2'
queue_name = 'seeds'
batch_size = 256
learning_rate = 1e-4