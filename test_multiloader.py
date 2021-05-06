from torch.utils.data import Dataset, DataLoader, get_worker_info
import time

class D(Dataset):
    def __init__(self):
        self.worker_id = None

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        print(self.worker_id, idx)
        return 0

def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    worker_id = worker_info.id
    dataset.worker_id = worker_info.id

d = D()

trainloader = DataLoader(d, batch_size = 8, shuffle = True, pin_memory = True, num_workers=8, worker_init_fn=worker_init_fn)
trainloader_iter = iter(trainloader)
a = next(trainloader_iter)
time.sleep(3)
b = next(trainloader_iter)
print(a.shape, b.shape)