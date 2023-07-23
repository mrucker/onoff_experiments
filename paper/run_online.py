import torch
import coba as cb

from corrals  import CorralSmoothIGW, CorralCappedIGW
from oracles  import LinearArgmin, ArgminPlusDispersion
from learners import LargeActionLearner
from rewards  import ScaledRewards

n_batches  = 10_000
batch_size = 8
n_processes = 1
out_file = "online.zip"

def gz_plus_100_times_t_to_3_4(gz,t):
    return gz + 100*t**(3/4)

def gz_plus_18_times_t(gz,t):
    return gz + 18*t

if n_processes > 1:
    torch.set_num_threads(1)

if __name__ == "__main__":

    cb.Environments.cache_dir(".coba_cache")

    lr, tz, eta = 0.02, 100, .3

    learners = []

    #SmoothIGW
    tmin, tmax, nalgos = 2, 2048, 12
    sampler = CorralSmoothIGW(eta=eta, gzero=1, gscale=gz_plus_100_times_t_to_3_4, tau_min=tmin, tau_max=tmax, nalgos=nalgos)
    learners.append(LargeActionLearner(sampler,ArgminPlusDispersion(argminblock=LinearArgmin()), lr, tz))

    #CappedIGW k_infty=24
    tmin, tmax, nalgos = 6, 1024, 12
    sampler = CorralCappedIGW(eta=eta, gzero=1, gscale=gz_plus_18_times_t, tau_min=tmin, tau_max=tmax, nalgos=nalgos, kappa_infty=24)
    learners.append(LargeActionLearner(sampler,ArgminPlusDispersion(argminblock=LinearArgmin()), lr, tz))

    #CappedIGW k_infty=4
    tmin, tmax, nalgos = 6, 1024, 12
    sampler = CorralCappedIGW(eta=eta, gzero=1, gscale=gz_plus_18_times_t, tau_min=tmin, tau_max=tmax, nalgos=nalgos, kappa_infty=4)
    learners.append(LargeActionLearner(sampler,ArgminPlusDispersion(argminblock=LinearArgmin()), lr, tz))

    #CappedIGW k_infty=2
    tmin, tmax, nalgos = 6, 1024, 12
    sampler = CorralCappedIGW(eta=eta, gzero=1, gscale=gz_plus_18_times_t, tau_min=tmin, tau_max=tmax, nalgos=nalgos, kappa_infty=2)
    learners.append(LargeActionLearner(sampler,ArgminPlusDispersion(argminblock=LinearArgmin()), lr, tz))

    datas = [41540,1187,44031,42225]
    tasks = [361088,361241,359937,361104,361255,233211,361089,361096,361251,361080,361094,361086,361242,361082,359939]
    envs  = cb.Environments.from_openml(data_id=datas)
    envs += cb.Environments.from_openml(task_id=tasks)
    envs += cb.Environments.from_openml(data_id=150,target="Elevation")

    envs   = envs.shuffle(n=30).take(n_batches*batch_size).impute(["median","mode"]).filter(ScaledRewards()).scale().batch(8)

    envs.logged(learners).save(out_file,processes=n_processes)
