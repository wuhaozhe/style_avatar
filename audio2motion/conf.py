def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_conf():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required = True)
    parser.add_argument("-it", "--iterations", help = "training iterations", default=200000, type=int)
    parser.add_argument("-b", "--batch_size", help="batch_size", default = 32, type=int)
    parser.add_argument("-m", "--multi_gpu", help="use multi gpu", default=False, type=str2bool)
    parser.add_argument("-lp", "--load_path", help="path of model", default=None, type=str)
    parser.add_argument("-lr", "--lr", help="learning rate", default=1e-3, type=float)
    parser.add_argument("-jp", "--json_path", help="config that written in json", default="./conf/ted_conf.json", type=str)
    conf = parser.parse_args()
    conf.milestones = [25000]
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if conf.load_path is None:
        conf.load = False
    else:
        conf.load = True

    conf.model_path = "./model/{}".format(conf.name)
    conf.log_path = "./log/{}".format(conf.name)
    conf.test_path = "./test/{}".format(conf.name)
    conf.fx = 50
    conf.fy = 25
    conf.xwin = 80
    conf.ywin = 32

    return conf

if __name__ == "__main__":
    conf = get_conf()
    print(conf)