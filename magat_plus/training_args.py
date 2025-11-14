import argparse


def add_training_args(parser):
    parser.add_argument("--validation_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--num_training_oe", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--imitation_learning_model", type=str, default="MAGATPlus")
    parser.add_argument("--cnn_mode", type=str, default="basic-CNN")
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--num_gnn_layers", type=int, default=3)
    parser.add_argument("--num_attention_heads", type=int, default=1)
    parser.add_argument("--attention_mode", type=str, default="GAT_modified")
    parser.add_argument("--edge_dim", type=int, default=None)
    parser.add_argument("--model_residuals", type=str, default=None)
    parser.add_argument(
        "--use_edge_weights", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--use_edge_attr", action=argparse.BooleanOptionalAction, default=False
    )

    parser.add_argument("--lr_start", type=float, default=1e-3)
    parser.add_argument("--lr_end", type=float, default=1e-6)
    parser.add_argument("--lr_scheduler", type=str, default="cosine-annealing")
    parser.add_argument("--num_epochs", type=int, default=200)

    parser.add_argument("--validation_every_epochs", type=int, default=4)
    parser.add_argument(
        "--run_online_expert", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--save_intmd_checkpoints", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")

    parser.add_argument(
        "--skip_validation", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--skip_validation_accuracy",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--project_name", type=str, default=None)
    parser.add_argument("--entity_name", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--model_seed", type=int, default=42)
    parser.add_argument("--initial_val_size", type=int, default=128)
    parser.add_argument("--threshold_val_success_rate", type=float, default=0.9)
    parser.add_argument("--num_run_oe", type=int, default=500)
    parser.add_argument("--run_oe_after", type=int, default=0)
    parser.add_argument(
        "--recursive_oe", action=argparse.BooleanOptionalAction, default=False
    )

    parser.add_argument(
        "--load_positions_separately",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--train_on_terminated_agents",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--run_expert_in_separate_fork",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--collision_shielding", type=str, default="naive")
    parser.add_argument("--action_sampling", type=str, default="deterministic")
    parser.add_argument("--action_sampling_temperature", type=float, default=1.0)

    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--module_residual", type=str, default=None)

    parser.add_argument("--use_edge_attr_for_messages", type=str, default=None)
    parser.add_argument("--edge_attr_processor", type=str, default=None)
    parser.add_argument("--max_runtime_oe", type=float, default=None)

    parser.add_argument("--pretrain_weights_path", type=str, default=None)

    return parser
