import itertools
import os
import sys
import warnings
import argparse
import numpy as np
import torch as T

from src.pipeline import GANPipeline, MLEPipeline
from src.scm.model_classes import XORModel, RoundModel
from src.scm.ctm import CTM
from src.scm.ncm import GAN_NCM, MLE_NCM
from src.run import RunTimeRunner
from src.ds import CTF
from src.metric.queries import get_query, get_experimental_variables, is_q_id_in_G

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

valid_pipelines = {
    "gan": GANPipeline,
    "mle": MLEPipeline,
}
valid_generators = {
    "ctm": CTM,
    "xor": XORModel,
    "round": RoundModel
}
architectures = {
    "gan": GAN_NCM,
    "mle": MLE_NCM,
}

gan_choices = {"vanilla", "bgan", "wgan", "wgangp"}
valid_graphs = {"backdoor", "bow", "frontdoor", "napkin", "simple", "bdm", "med", "expl", "double_bow", "iv", "bad_fd",
                "extended_bow", "bad_m", "m",
                "zid_a", "zid_b", "zid_c",
                "gid_a", "gid_b", "gid_c", "gid_d",
                "med_c1", "med_c2",
                "expl_xm", "expl_xm_dox", "expl_xy", "expl_dox", "expl_xy_dox", "expl_my", "expl_my_dox"}

graph_sets = {
    "all": {"backdoor", "bow", "frontdoor", "napkin", "simple", "med", "expl", "zid_a", "zid_b", "zid_c",
            "gid_a", "gid_b", "gid_c", "gid_d"},
    "standard": {"backdoor", "bow", "frontdoor", "napkin", "iv", "extended_bow", "m", "bad_m", "bdm"},
    "zid": {"zid_a", "zid_b", "zid_c"},
    "gid": {"gid_a", "gid_b", "gid_c", "gid_d"},
    "expl_set": {"expl", "expl_dox", "expl_xm", "expl_xm_dox", "expl_xy", "expl_xy_dox", "expl_my", "expl_my_dox"}
}

valid_queries = {"ate", "ett", "nde", "ctfde"}

parser = argparse.ArgumentParser(description="Basic Runner")
parser.add_argument('name', help="name of the experiment")

parser.add_argument('--gen', default="ctm", help="data generating model (default: ctm)")
parser.add_argument('--lr', type=float, default=4e-3, help="optimizer learning rate (default: 4e-3)")
parser.add_argument('--data-bs', type=int, default=1000, help="batch size of data (default: 1000)")
parser.add_argument('--ncm-bs', type=int, default=1000, help="batch size of NCM samples (default: 1000)")
parser.add_argument('--h-layers', type=int, default=2, help="number of hidden layers (default: 2)")
parser.add_argument('--h-size', type=int, default=64, help="neural network hidden layer size (default: 128)")
parser.add_argument('--u-size', type=int, default=1, help="dimensionality of U variables (default: 1)")
parser.add_argument('--neural-pu', action="store_true", help="use neural parameters in U distributions")
parser.add_argument('--layer-norm', action="store_true", help="set flag to use layer norm", default=True)

parser.add_argument('--gan-mode', default="wgan", help="GAN loss function (default: vanilla)")
parser.add_argument('--d-iters', type=int, default=1,
                    help="number of discriminator iterations per generator iteration (default: 1)")
parser.add_argument('--grad-clamp', type=float, default=0.01,
                    help="value for clamping gradients in WGAN (default: 0.01)")
parser.add_argument('--gp-weight', type=float, default=10.0,
                    help="regularization constant for gradient penalty in WGAN-GP (default: 10.0)")

parser.add_argument('--full-batch', action="store_true", help="use n as the batch size")

parser.add_argument('--regions', type=int, default=20, help="number of regions for CTM (default: 20)")
parser.add_argument('--scale-regions', action="store_true", help="scale regions by C-cliques in CTM")
parser.add_argument('--gen-bs', type=int, default=10000, help="batch size of ctm data generation (default: 10000)")
parser.add_argument('--no-positivity', action="store_true", help="does not enforce positivity for ID experiments")

parser.add_argument('--query-track', default="ate", help="choice of query to track")
parser.add_argument('--id-reruns', '-r', type=int, default=1,
                    help="hypothesis testing trials for ID experiments (default: 1)")
parser.add_argument('--max-query-iters', type=int, default=3000, help="number of ID iterations (default: 3000)")
parser.add_argument('--max-lambda', type=float, default=1.0, help="regularization constant start (default: 1)")
parser.add_argument('--min-lambda', type=float, default=0.001, help="regularization constant end (default: 1e-3)")
parser.add_argument('--mc-sample-size', type=int, default=10000,
                    help="sample size for query optimization (default: 10000)")
parser.add_argument('--single-disc', action="store_true", help="use one discriminator")
parser.add_argument('--gen-sigmoid', action="store_true", help="use sigmoids in generator")
parser.add_argument('--perturb-sd', type=float, default=0.1,
                    help="standard deviation of distribution for perturbing data")

parser.add_argument('--graph', '-G', default="expl", help="name of preset graph")
parser.add_argument('--n-trials', '-t', type=int, default=4, help="number of trials")
parser.add_argument('--n-samples', '-n', type=int, default=10000, help="number of samples (default: 10000)")
parser.add_argument('--dim', '-d', type=int, default=1, help="dimensionality of variables (default: 1)")
parser.add_argument('--gpu', help="GPU to use")

parser.add_argument('--verbose', action="store_true", help="print more information")


args = parser.parse_args()

gen_choice = args.gen.lower()
graph_choice = args.graph.lower()
gan_choice = args.gan_mode.lower()
query_track = args.query_track

if query_track is not None:
    query_track = query_track.lower()

assert gen_choice in valid_generators
assert graph_choice in valid_graphs or graph_choice in graph_sets
assert gan_choice in gan_choices
assert query_track is None or query_track in valid_queries

dat_model = valid_generators[gen_choice]
gpu_used = 0 if args.gpu is None else [int(args.gpu)]

arg_data_bs = args.data_bs
arg_ncm_bs = args.ncm_bs

hyperparams = {
    'lr': args.lr,
    'data-bs': args.data_bs,
    'ncm-bs': args.ncm_bs,
    'h-layers': args.h_layers,
    'h-size': args.h_size,
    'u-size': args.u_size,
    'neural-pu': args.neural_pu,
    'layer-norm': args.layer_norm,
    'regions': args.regions,
    'c2-scale': 2.0 if args.scale_regions else 1.0,
    'gen-bs': args.gen_bs,
    'gan-mode': gan_choice,
    'd-iters': args.d_iters,
    'grad-clamp': args.grad_clamp,
    'gp-weight': args.gp_weight,
    'query-track': query_track,
    'id-reruns': args.id_reruns,
    'max-query-iters': args.max_query_iters,
    'min-lambda': args.min_lambda,
    'max-lambda': args.max_lambda,
    'mc-sample-size': args.mc_sample_size,
    'single-disc': args.single_disc,
    'gen-sigmoid': args.gen_sigmoid,
    'perturb-sd': args.perturb_sd,
    'full-batch': args.full_batch,
    'positivity': not args.no_positivity
}

graph = args.graph
do_var_list = get_experimental_variables(graph)
eval_query, opt_query = get_query(graph, query_track)

hyperparams['do-var-list'] = do_var_list
hyperparams['eval-query'] = eval_query

d_list = [1, 2, 3, 4, 5, 6, 7]
n = int(args.n_samples)
for d in d_list:
    for i in range(args.n_trials):
        for pipeline_choice in ['gan', 'mle']:
            print("#############################################################")
            print('pipeline is ', pipeline_choice)
            print("#############################################################")
            if pipeline_choice == "gan":
                hyperparams['lr'] = 0.00002
                hyperparams["u-size"] = 2
                hyperparams["h-size"] = 128
                hyperparams['full-batch'] = False
                hyperparams['data-bs'] = hyperparams['data-bs'] * hyperparams['d-iters']
                hyperparams["data-bs"] = min(arg_data_bs, n)
                hyperparams["ncm-bs"] = min(arg_ncm_bs, n)
                if arg_data_bs == -1:
                    hyperparams["data-bs"] = n // 10
                if arg_ncm_bs == -1:
                    hyperparams["ncm-bs"] = n // 10

            if pipeline_choice == 'mle':
                hyperparams['lr'] = 4e-3
                hyperparams['full-batch'] = True
                hyperparams["data-bs"] = n
                hyperparams['ncm-bs'] = n
                hyperparams["u-size"] = 1
                hyperparams["h-size"] = 128
            hyperparams["pipeline_choice"] = pipeline_choice
            pipeline = valid_pipelines[pipeline_choice]
            dat_model = valid_generators[gen_choice]
            ncm_model = architectures[pipeline_choice]

            cg_file = "dat/cg/{}.cg".format(graph)
            try:
                runner = RunTimeRunner(pipeline, dat_model, ncm_model)
                runner.run(args.name, cg_file, n, d, i, hyperparams=hyperparams, gpu=gpu_used, verbose=args.verbose)
            except Exception as e:
                print(e)
                print('[failed]', i, args.name)
                raise

