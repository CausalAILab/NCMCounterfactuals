import os
from tempfile import NamedTemporaryFile
from contextlib import contextmanager


class BaseRunner:
    def __init__(self, pipeline, dat_model, ncm_model):
        self.pipeline = pipeline
        self.pipeline_name = pipeline.__name__
        self.dat_model = dat_model
        self.dat_model_name = dat_model.__name__
        self.ncm_model = ncm_model
        self.ncm_model_name = ncm_model.__name__

    @contextmanager
    def lock(self, file, lockinfo):  # attempt to acquire a file lock; yield whether or not lock was acquired
        os.makedirs(os.path.dirname(file), exist_ok=True)
        os.makedirs('tmp/', exist_ok=True)
        with NamedTemporaryFile(dir='tmp/') as tmp:
            try:
                os.link(tmp.name, file)
                acquired_lock = True
            except FileExistsError:
                acquired_lock = os.stat(tmp.name).st_nlink == 2
        if acquired_lock:
            with open(file, 'w') as fp:
                fp.write(lockinfo)
            try:
                yield True
            finally:
                try:
                    os.remove(file)
                except FileNotFoundError:
                    pass
        else:
            yield False

    def get_key(self, cg_file, n, dim, trial_index):
        graph = cg_file.split('/')[-1].split('.')[0]
        return ('gen=%s-graph=%s-n_samples=%s-dim=%s-trial_index=%s'
                % (self.dat_model_name, graph, n, dim, trial_index))

    def run(self, exp_name, cg_file, n, dim, trial_index, hyperparams=None, gpu=None,
            lockinfo=os.environ.get('SLURM_JOB_ID', ''), verbose=False):
        raise NotImplementedError()
