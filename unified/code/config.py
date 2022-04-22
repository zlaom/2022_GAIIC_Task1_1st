import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument("--seed", default=2022, type=int)
    parser.add_argument(
        "--reset_type",
        default="new",
        choices=["new", "repeat", "cycle"],
        type=str,
    )
    parser.add_argument("--job_seq_num", default=50, type=int)
    parser.add_argument("--max_time", default=200, type=int)
    parser.add_argument("--max_end_time", default=250, type=int)
    parser.add_argument("--max_job_num", default=20, type=int)
    parser.add_argument("--max_res_req", default=8, type=int)
    parser.add_argument("--max_job_len", default=15, type=int)
    parser.add_argument("--job_small_rate", default=0.6, type=float)
    parser.add_argument(
        "--level_job_num",
        # default=[5, 5, 20, 20, 10, 10, 20, 20, 5, 5],
        # default=[10, 10, 8, 4, 2, 10, 10, 8, 4, 2],
        default=[2, 4, 8, 8, 6, 4, 6, 8, 8, 4, 2],
        type=list,
    )
    # parser.add_argument(
    #     "--level_job_long_rate",
    #     default=[0.9, 0.6, 0.5, 0.6, 0.7, 0.5, 0.6, 0.7, 0.8, 0.8, 0.9],
    #     type=list,
    # )
    parser.add_argument("--sleep_delay", default=3, type=int)
    parser.add_argument("--pre_allocate_time", default=1, type=int)
    parser.add_argument("--crisis_rate", default=0.8, type=float)
    parser.add_argument("--max_crisis_rate", default=0.8, type=float)
    parser.add_argument("--crisis_time", default=5, type=int)
    parser.add_argument("--res_num", default=2, type=int)
    parser.add_argument("--res_power_weight", default=[0.7, 0.3], type=list)
    parser.add_argument("--res_std_weight", default=[0.7, 0.3], type=list)
    parser.add_argument("--res_var_weight", default=[0.7, 0.3], type=list)
    parser.add_argument("--base_power", default=0.5, type=float)
    parser.add_argument("--res_capacity", default=10, type=int)
    parser.add_argument("--machine_num", default=20, type=int)
    parser.add_argument("--max_expand_num", default=5, type=int)
    parser.add_argument("--buffer_size", default=50, type=int)
    parser.add_argument("--timeline_size", default=20, type=int)
    parser.add_argument("--job_color_num", default=40, type=int)
    parser.add_argument(
        "--job_generate",
        default="level_bi_model",
        choices=[
            "uniform",
            "level_uniform",
            "level_bi_model",
        ],
        type=str,
    )
    parser.add_argument(
        "--obs_represent",
        default="timeline",
        choices=["image", "timeline"],
        type=str,
    )
    parser.add_argument(
        "--end_mode",
        default="all_allocate",
        choices=[
            "all_allocate",
            "all_done",
            "max_time",
        ],
        type=str,
    )
    parser.add_argument("--reward_scale", default=1, type=int)
    parser.add_argument(
        "--reward_type",
        default="run_time_and_var",
        choices=[
            "machine_run_num",
            "machine_power",
            "job_slowdown",
            "curr_res_rate",
            "res_std",
            "res_var",
            "run_time_and_var",
        ],
        type=str,
    )

    # ppo agent
    parser.add_argument("--num_games", default=200000, type=int)
    parser.add_argument(
        "--save_path",
        # default="output/new/res_var/mask_ppov1_must_allocate_must_expand_20_pre_1/run02",
        # default="output/new/res_var/mask_ppov1_must_allocate_must_expand_20_pre_1/run01",
        # default="output/new/geneic/mask_ga_optimize_var/run03",
        # default="output/new/geneic/mask_ga_optimize_var/run01",
        default="output/new/geneic/mask_ga_optimize_var_runtime/run02",
        type=str,
    )
    parser.add_argument(
        "--update_timestep",
        default=128,
        type=int,
        help="update policy every n timesteps",
    )
    parser.add_argument(
        "--K_epochs",
        default=5,
        type=int,
        help="update policy for K epochs in one PPO update",
    )
    parser.add_argument(
        "--eps_clip",
        default=0.2,
        type=float,
        help="clip parameter for PPO",
    )
    parser.add_argument(
        "--gamma",
        default=0.99,
        type=float,
        help="clip parameter for PPO",
    )
    parser.add_argument(
        "--lr_actor",
        default=1e-3,
        type=float,
        help="clip parameter for PPO",
    )
    parser.add_argument(
        "--lr_critic",
        default=5e-4,
        type=float,
        help="clip parameter for PPO",
    )
    parser.add_argument(
        "--ppo_hidden_dim",
        default=32,
        type=int,
        help="clip parameter for PPO",
    )
    parser.add_argument(
        "--use_job_actor",
        default=True,
        type=bool,
    )
    parser.add_argument(
        "--use_expand_actor",
        default=True,
        type=bool,
    )

    # genetic
    parser.add_argument("--ga_parent_size", default=25, type=int)
    parser.add_argument("--ga_children_size", default=25, type=int)
    parser.add_argument("--ga_mutate_rate", default=0.15, type=float)
    parser.add_argument("--ga_choice", default="generate", type=str)

    args, unknown = parser.parse_known_args()
    return args
