from torch.utils.data import get_worker_info
import random

def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    worker_id = worker_info.id
    dataset.worker_id = worker_info.id

def gen_randomwalk_list(in_list, target_len, min_walk_len = 20):
    in_list_pad = [in_list[0]] + in_list + [in_list[-1]]
    current_idx = 1
    total_len = 0
    direction = False
    target_list = []
    while total_len < target_len:
        next_len = random.randint(min_walk_len, len(in_list))
        for _ in range(next_len):
            target_list.append(in_list_pad[current_idx])
            if direction == False:
                current_idx += 1
                if current_idx == len(in_list_pad) - 1:
                    direction = True
            else:
                current_idx -= 1
                if current_idx == 0:
                    direction = False
        if random.randint(0, 1) == 1 and current_idx != 0 and current_idx != len(in_list_pad) - 1:
            direction = not direction

        total_len += next_len

    return target_list[:target_len]


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

if __name__ == "__main__":
    print(gen_randomwalk_list(list(range(100)), 500))