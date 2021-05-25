from torch.utils.data import get_worker_info

def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    worker_id = worker_info.id
    dataset.worker_id = worker_info.id

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None

    def update(self, val):
        if self.val is None:
            self.val = val
        else:
            self.val = self.val * 0.9 + val * 0.1

    def __call__(self):
        return self.val