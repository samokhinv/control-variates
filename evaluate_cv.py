def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv_lr', default=1e-6, type=float)
    parser.add_argument('--n_cv_iter', default=100, type=int)
    parser.add_argument('--batch_size', default=-1, type=int)
    parser.add_argument('--input_dim', default=784, type=int)
    parser.add_argument('--width', type=int, default=100)
    parser.add_argument('--depth', type=int, default=0)
    parser.add_argument('--output_dim', type=int, default=2)
    parser.add_argument('--samples_path', type=str, required=True)
    parser.add_argument('--cv_type', type=str, choices=['const', 'mlp'], default='const')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--data_dir', type=str, default='../data/mnist')
    parser.add_argument('--classes', type=int, nargs='+', default=[3, 5])
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    return args