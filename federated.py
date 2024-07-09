from datetime import datetime
from pathlib import Path
import json
import docker
import torch
import seaborn as sns
try:
    import sys
    sys.path.append('/mnt/ssd/git-repos/fedbiomed')
    from fedbiomed.researcher.environ import environ
    from fedbiomed.researcher.experiment import Experiment
    from fedbiomed.researcher.aggregators.fedavg import FedAverage
except:
    pass

NAME_TO_NODE_ID = {
    'heidelberg': 'node_24eb98e6-daca-41a5-8ba7-920ddefa783d', # 'node_b78aa472-ebb1-45a3-bd52-7e6a5ead6be4',
    'muenster': 'node_ac65d070-f069-4125-8520-a6a3b70d53a6',
    'frankfurt': 'node_80377511-c70c-49d6-879d-2e10ae184b64',  # 'node_31b29d9e-ba67-41cc-a47c-41ea8964b401',
    'greifswald': 'node_4d52237f-9325-4e99-b76e-91ca543832ad',
    'hamburg': 'node_c20a9b09-e37b-42ab-a3db-5f66321831a7', # 'node_b4818caa-abc1-461d-a038-95b8974a38fb',
    'munich': 'node_1c72b46e-4065-42eb-a6a3-ec013fc0a723',
    'goettingen': 'node_623abeb5-6468-4e85-a6a4-228be1301bd8', # 'node_a6a8b25d-57b5-4b30-85dc', # 'node_e34258ff-7bb3-4197-ba97-af5c94dcc030',
    'berlin': 'node_4716d7f1-6532-494b-b4f0-4d47afe8132f',
    'wuerzburg': ''
}
cp = sns.color_palette()
COLORS_long = {
    'heidelberg': cp[0],
    'muenster': cp[1],
    'goettingen': cp[2],
    'munich': cp[3],
    'frankfurt': cp[4],
    'hamburg': cp[5],
    'berlin': cp[6],
    'greifswald': cp[7],
    'fed': cp[8],
    'kd': cp[9],
    'wuerzburg': ''
}
COLORS_short = {
    'HD': cp[0],
    'MS': cp[1],
    'GOE': cp[2],
    'M': cp[3],
    'F': cp[4],
    'HH': cp[5],
    'B': cp[6],
    'GFS': cp[7],
    'Fed': cp[8],
    r'$\mathrm{KD}$': cp[9],
    'wuerzburg': ''
}

def run_federated_training_from_breakpoint(bkpt):
    exp = Experiment.load_breakpoint(bkpt) ## experiments/federated/train_nnUNet_hinge_points_frankfurt_muenster_goettingen_munich_heidelberg/1702507709.317014/breakpoint_0004
    exp.run()
    torch.save(exp.aggregated_params()[num_rounds - 1]['params'], f'{experimentation_folder}/final_ckpt.pt')

def run_federated_training(
        nodes,
        training_plan,
        tags,
        num_rounds,
        training_args,
        experimentation_folder=None,
        use_secagg=False,
        secagg_timeout=120,
        model_args={}
):
    exp = Experiment(
        nodes=nodes,
        tags=tags,
        model_args=model_args,
        training_plan_class=training_plan,
        training_args=training_args,
        round_limit=num_rounds,
        aggregator=FedAverage(),
        tensorboard=True,
        experimentation_folder=experimentation_folder.split('/')[-1],
        node_selection_strategy=None,
        use_secagg=use_secagg, # or custom SecureAggregation(active=<bool>, clipping_range=<int>, timeout=<int>)
        secagg_timeout=secagg_timeout,
        save_breakpoints=True
    )
    exp.run()
    # if training_args['test_ratio'] < 1.0:
    torch.save(exp.aggregated_params()[num_rounds - 1]['params'], f'{experimentation_folder}/final_ckpt.pt')

def run_federated_testing(
        nodes,
        tags,
        num_rounds,
        training_args,
        training_plan,
        experimentation_folder=None,
        model_args={}
):
    exp = Experiment(
        nodes=nodes,
        tags=tags,
        model_args=model_args,
        training_plan_class=training_plan,
        training_args=training_args,
        round_limit=num_rounds,
        aggregator=FedAverage(),
        tensorboard=True,
        experimentation_folder=experimentation_folder.split('/')[-1],
    )
    if exp._global_model is None:
        exp._global_model = exp._job.training_plan.get_model_params()
    exp._aggregator.set_training_plan_type(exp._job.training_plan.type())
    exp._job.nodes = exp._node_selection_strategy.sample_nodes(exp._round_current)
    exp._aggregator.check_values(
        n_updates=exp._training_args.get('num_updates'),
        training_plan=exp._job.training_plan
    )
    aggr_args_thr_msg, aggr_args_thr_file = exp._aggregator.create_aggregator_args(exp._global_model, exp._job._nodes)
    exp._job.start_nodes_training_round(
        round=exp._round_current,
        aggregator_args_thr_msg=aggr_args_thr_msg,
        aggregator_args_thr_files=aggr_args_thr_file,
        do_training=False
    )

def federated_experiment(args, training_plan, model_args={}, breakpoint=None):
    if not args.local:
        import os
        environ['MQTT_BROKER'] = 'mqtt.fed-learning.org' # '129.206.7.138'
        environ['MQTT_BROKER_PORT'] = 80 # 1883
        #os.environ['UPLOADS_URL'] = 'http://129.206.7.138:8844/upload/'
        environ['UPLOADS_URL'] = 'https://uploads.fed-learning.org/upload/'
        environ['MQTT_LOGIN_ROUTE'] = 'https://develop.fed-learning.org:443/login'
        os.environ['MQTT_BROKER_TRANSPORT_PROTOCOL'] = 'websockets'

    if breakpoint is not None:
        bkpt_path = Path(breakpoint)
        environ['EXPERIMENTS_DIR'] = str(bkpt_path.parents[1]) # f'./experiments/federated/{args.exp_name}'
        environ['TENSORBOARD_RESULTS_DIR'] = str(bkpt_path.parents[0] / 'tb_logs') # f'{experimentation_folder}/tb_logs'
        run_federated_training_from_breakpoint(breakpoint)
        return
    
    ts = str(datetime.now().timestamp())
    experimentation_folder = f'./experiments/federated/{args.exp_name}/{ts}'
    environ['EXPERIMENTS_DIR'] = f'./experiments/federated/{args.exp_name}'
    
    environ['TENSORBOARD_RESULTS_DIR'] = f'{experimentation_folder}/tb_logs'
    _ef = Path(experimentation_folder)
    _ef.mkdir(parents=True)
    client = docker.from_env()
    tb_containers = [c for c in client.containers.list() if 'fedbiomed-tensorboard' in c.name]
    if len(tb_containers) > 1:
        print('Trying to stop and remove more than one container. Manual interaction needed!')
        sys.exit()
    for c in tb_containers:
        c.stop()
        c.remove()
    tb_dir = (_ef / 'tb_logs')#.absolute()
    tb_dir.mkdir()
    # tb_container = client.containers.run(
    #     'schafo/tensorboard',
    #     name=f'fedbiomed-tensorboard-{args.exp_name}',
    #     volumes={str(tb_dir.absolute()): {'bind': '/app/runs/', 'mode': 'ro'}},
    #     ports={6006: 6006},
    #     detach=True
    # )
    
    if args.ckpt is not None:
        os.environ['CKPT'] = args.ckpt

    nodes = None
    if args.locations is not None:
        nodes = [NAME_TO_NODE_ID[l] for l in args.locations]
    # with open(args.config, 'r') as f:
    #     config = json.load(f)
    config = {
        "num_rounds": args.num_rounds,
        "training_args": {
            "use_gpu": True,
            "batch_size": args.batch_size, 
            "epochs": args.epochs,
            "dry_run": False,
            "log_interval": 1,
            "test_ratio" : args.test_ratio,
            "test_on_global_updates": args.test_on_global_updates,
            "test_on_local_updates": args.test_on_local_updates
        }
    }

    if args.batch_maxnum is not None:
        config['training_args']['batch_maxnum'] = args.batch_maxnum

    config['locations'] = args.locations
    with open(_ef / 'config.json', 'w') as f:
        json.dump(config, f)

    if args.mode == 'train':
        run_federated_training(
            nodes=nodes,
            training_plan=training_plan,
            tags=args.tags,
            num_rounds=config['num_rounds'],
            training_args=config['training_args'],
            experimentation_folder=experimentation_folder,
            model_args=model_args,
            use_secagg=config.get('use_secagg', False),
            secagg_timeout=config.get('secagg_timeout', 120)
        )
    elif args.mode == 'test':
        run_federated_testing(
            nodes=nodes,
            training_plan=training_plan,
            tags=args.tags,
            num_rounds=config['num_rounds'],
            training_args=config['training_args'],
            experimentation_folder=experimentation_folder,
            model_args=model_args
        )