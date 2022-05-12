import copy
import os.path

import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib import cm

datafiles = [ "94:64:24:9b:46:50-IllinoisNet.data",
              "94:64:24:99:d7:62-IllinoisNet_Guest.data",
              "94:64:24:9b:b7:30-IllinoisNet.data" ]
datas = { datafile: np.load(datafile, allow_pickle=True) for datafile in datafiles }
models = { k: scipy.interpolate.LinearNDInterpolator(data[:, :2], data[:, 2]) for k, data in datas.items() }

raw_xs = datas[datafiles[0]][:, 0]
raw_ys = datas[datafiles[0]][:, 1]
mid_x = (np.min(raw_xs) + np.max(raw_xs)) / 2
mid_y = (np.min(raw_ys) + np.max(raw_ys)) / 2

np.random.seed(0)

sharpness = 1
def evaluate(params, x, y):
    x0, y0, A = params
    r = (x - x0)**2 + (y-y0)**2
    return A / (r + sharpness)

def grad(params, x, y):
    """
    computes derivative of zvals w.r.t. x0, y0, A.
    """
    x0, y0, A = params

    zvals = evaluate(params, x, y)
    x_c = 2*x
    x_a = x**2 + (y-y0)**2 + sharpness
    x_deriv = A * (-2*x0 + x_c) / ((x0**2 - x_c*x + x_a)**2)
    y_c = 2*y
    y_a = y**2 + (x-x0)**2 + sharpness
    y_deriv = A * (-2*y0 + y_c) / ((y0**2 - y_c*y + y_a)**2)
    A_deriv = zvals
    return x_deriv, y_deriv, A_deriv

lr = 1
fit_trials = 200
def fit(xs, ys, zs, n_particles):
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    A0 = max(zs)

    x = np.random.uniform(min_x, max_x, (n_particles, 1))
    y = np.random.uniform(min_y, max_y, (n_particles, 1))
    A = np.ones((n_particles, 1)) * A0
    for i in range(5000):
        zvals = evaluate((x, y, A), xs, ys)
        err_deriv = 1000*(zvals - zs)*2 / len(zs)
        x_grad, y_grad, A_grad = grad((x, y, A), xs, ys)
        x -= lr * np.sum(x_grad * err_deriv, axis=1).reshape((-1, 1))
        y -= lr * np.sum(y_grad * err_deriv, axis=1).reshape((-1, 1))
        A -= lr * np.sum(A_grad * err_deriv, axis=1).reshape((-1, 1))

    zvals = evaluate((x, y, A), xs, ys)
    err = (1000*(zvals - zs))**2 / len(zs)
    err_val = np.sum(err, axis=1)
    best = np.argmin(err_val)
    return err_val[best], (x[best, 0], y[best, 0], A[best, 0])

fit_model_path = "fit_models.npy"
if os.path.exists(fit_model_path):
    print("Using saved models")
    fit_models = np.load(fit_model_path, allow_pickle=True).item(0)
else:
    print("Fitting models... ", end="", flush=True)
    fit_models = { k: fit(data[:, 0], data[:, 1], data[:, 2], fit_trials)[1] for k, data in datas.items() }
    np.array(fit_models).dump(fit_model_path)
    print("done.")

true_fig, true_axs = plt.subplots(2, 2, subplot_kw={"projection": "3d"})
true_axs = true_axs.flatten()

test_fig, test_axs = plt.subplots(2, 2, subplot_kw={"projection": "3d"})
test_axs = test_axs.flatten()

VEL_STDEV = 0.1
POS_STDEV = 0.1
GRID_STEPS = 21
GRID_RADIUS = 1

def gaussian(x, mean, stdev):
    """
    Calculate value of ND gaussian distribution with mean and stdev.
    """
    xvec = x - mean
    return np.exp(-xvec**2 / (2 * stdev**2)) / (np.sqrt(2*np.pi) * stdev)

def likelihood(params, x, y, observed, stdev):
    zval = evaluate(params, x, y)
    return gaussian(observed, zval, stdev)

def score_particle(env, particle_pos, observed):
    center = np.mean(env['pos'], axis=0)
    xs = np.linspace(center[0] - GRID_RADIUS, center[0] + GRID_RADIUS, GRID_STEPS)
    ys = np.linspace(center[1] - GRID_RADIUS, center[1] + GRID_RADIUS, GRID_STEPS)
    X, Y = np.meshgrid(xs, ys)
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    score = 0
    for key in observed:
        params = env['parameters'][key]
        # HACK assume noise proportional to signal lol...
        Z = likelihood(params, X, Y, observed[key], params[2])
        X_off = X - particle_pos[0]
        Y_off = Y - particle_pos[1]
        pos_dist = gaussian(np.sqrt(X_off**2 + Y_off**2), 0, POS_STDEV)
        score += np.log(np.dot(pos_dist, Z) + 1e-3)
    return score

def score_env(env, observed):
    pos = env['pos']
    vel = env['vel']
    scores = np.zeros(len(pos))
    center = np.mean(pos, axis=0)
    xs = np.linspace(center[0] - GRID_RADIUS, center[0] + GRID_RADIUS, GRID_STEPS)
    ys = np.linspace(center[1] - GRID_RADIUS, center[1] + GRID_RADIUS, GRID_STEPS)
    X, Y = np.meshgrid(xs, ys)
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    for key in observed:
        params = env['parameters'][key]
        # HACK assume noise proportional to signal lol...
        Z = likelihood(params, X, Y, observed[key], params[2])
        for i, particle_pos in enumerate(pos):
            X_off = X - particle_pos[0]
            Y_off = Y - particle_pos[1]
            pos_dist = gaussian(np.sqrt(X_off**2 + Y_off**2), 0, POS_STDEV)
            scores[i] += np.log(np.dot(pos_dist, Z) + 1e-3)
    return scores

def repopulate_particles(env, scores):
    sort_indices = np.argsort(scores)
    num_particles = len(scores)
    half_num = num_particles // 2
    quarter_num = half_num // 2
    new_pos = np.empty(env['pos'].shape)
    new_vel = np.empty(env['vel'].shape)
    save_indices = np.random.choice(sort_indices[:half_num], size=quarter_num, replace=False)
    new_pos[:half_num, :] = env['pos'][sort_indices, :][half_num:, :]
    new_vel[:half_num, :] = env['vel'][sort_indices, :][half_num:, :]
    new_pos[half_num:half_num+quarter_num, :] = env['pos'][save_indices, :]
    new_vel[half_num:half_num+quarter_num, :] = env['pos'][save_indices, :]
    new_pos[half_num+quarter_num:, :] = new_pos[:quarter_num, :]
    new_vel[half_num+quarter_num:, :] = new_vel[:quarter_num, :]
    env['pos'] = new_pos
    env['vel'] = new_vel

def purge_envs(envs, pos_pred, scores, measurement, n_particles):
    sort_indices = np.argsort(scores)
    num_particles = len(scores)
    half_num = num_particles // 2
    quarter_num = half_num // 2
    new_envs = []
    save_indices = np.random.choice(sort_indices[:half_num], size=quarter_num, replace=False)
    for i in range(half_num, len(scores)):
        new_envs.append(envs[sort_indices[i]])
    for saved_index in save_indices:
        new_envs.append(envs[saved_index])
    
    new_envs += init_particles(pos_pred, [0, 0], measurement, quarter_num, n_particles)
    return new_envs

def motion_model(env, disp):
    """
    Velocity delta is gaussian centered at 0.
    Position delta is gaussian centered at velocity.
    """
    env['pos'] += env['vel']
    #env['pos'] += disp
    env['pos'] += np.random.normal(0, POS_STDEV, env['pos'].shape)
    env['vel'] += np.random.normal(0, VEL_STDEV, env['vel'].shape)

def init_particles(pos, vel, measurement, env_particles, pos_particles):
    particles = []
    for i in range(env_particles):
        centers = { key: pos + np.random.uniform(-2, 2, 2) for key in measurement }
        param_sets = {}
        for key, val in measurement.items():
            x, y = centers[key]
            normalize_val = evaluate((x, y, 1), *pos)
            param_sets[key] = [x, y, val / normalize_val]
        particles.append({ 'pos': np.ones((pos_particles, 1)) * pos,
                           'vel': np.random.normal(0, VEL_STDEV, (pos_particles, 2)),
                           'parameters': param_sets })
    return particles

def measure(pos_real):
    return { f: models[f](pos_real) for f in datafiles }


def plot_env(env, pos_real, pos_pred):
    xs = np.linspace(pos_real[0] - GRID_RADIUS, pos_real[0] + GRID_RADIUS, GRID_STEPS)
    ys = np.linspace(pos_real[1] - GRID_RADIUS, pos_real[1] + GRID_RADIUS, GRID_STEPS)
    X, Y = np.meshgrid(xs, ys)
    _X, _Y = X.reshape(-1), Y.reshape(-1)
    for ax, (name, model) in zip(true_axs, models.items()):
        ax.clear()
        C = model(_X, _Y).reshape(X.shape)
        vmin = np.nanmin(C)
        vmax = np.nanmax(C)
        vdiff = vmax - vmin
        ax.plot_surface(X, Y, C, vmin=vmin, vmax=vmax, cmap=cm.coolwarm)
        ax.plot((pos_real[0], pos_real[0]), (pos_real[1], pos_real[1]), (vmin-0.1*vdiff, vmax+0.1*vdiff), color='blue')
        #ax.plot((pos_pred[0], pos_pred[0]), (pos_pred[1], pos_pred[1]), (vmin-0.1*vdiff, vmax+0.1*vdiff), color='red')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(name)

    xs = np.linspace(pos_pred[0] - GRID_RADIUS, pos_pred[0] + GRID_RADIUS, GRID_STEPS)
    ys = np.linspace(pos_pred[1] - GRID_RADIUS, pos_pred[1] + GRID_RADIUS, GRID_STEPS)
    X, Y = np.meshgrid(xs, ys)
    _X, _Y = X.reshape(-1), Y.reshape(-1)
    for ax, (name, params) in zip(test_axs, env['parameters'].items()):
        ax.clear()
        C = evaluate(params, _X, _Y).reshape(X.shape)
        vmin = np.nanmin(C)
        vmax = np.nanmax(C)
        vdiff = vmax - vmin
        ax.plot_surface(X, Y, C, vmin=vmin, vmax=vmax, cmap=cm.coolwarm)
        #ax.plot((pos_real[0], pos_real[0]), (pos_real[1], pos_real[1]), (vmin-0.1*vdiff, vmax+0.1*vdiff), color='blue')
        ax.plot((pos_pred[0], pos_pred[0]), (pos_pred[1], pos_pred[1]), (vmin-0.1*vdiff, vmax+0.1*vdiff), color='red')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(name)
    plt.show(block=False)
    plt.pause(0.05)


n_env = 400
n_particles = 64
pos_real = np.array([mid_x, mid_y])
pos_sim = np.array(pos_real)
vel_sim = np.array([0, 0])

#measurement = measure(pos_real)
#envs = init_particles(pos_sim, vel_sim, measurement, n_env, n_particles)
#s_env = envs[0]
#print(s_env['parameters'])
#def update_envs(disp):
#    global envs
#    measurement = measure(pos_real)
#    env_scores = np.empty(len(envs))
#    for i, env in enumerate(envs):
#        motion_model(env, disp)
#        scores = score_env(env, measurement)
#        repopulate_particles(env, scores)
#        env_scores[i] = max(scores)
#        if i % 20 == 0:
#            print(i)
#    env_ranking = np.argsort(env_scores)
#    best_env = envs[env_ranking[-1]]
#    print(best_env)
#    pos_pred = np.mean(best_env['pos'], axis=0)
#    envs = purge_envs(envs, pos_pred, env_scores, measurement, n_particles)
#    print(pos_pred)
#    plot_env(best_env, pos_real, pos_pred)
#
#for i in range(10):
#    disp = [0, 0.1]
#    pos_real += disp
#    update_envs(disp)
#    print(pos_real)
#    input()
#
#for i in range(10):
#    disp = [0, -0.1]
#    pos_real += disp
#    update_envs(disp)
#    print(pos_real)
#    input()

# ---------------------------------------
# Testing fitted model motion prediction.
# ---------------------------------------
test_env = {
            'pos': np.ones((n_particles, 1)) * pos_real,
            'vel': np.random.normal(0, VEL_STDEV, (n_particles, 2)),
            'parameters': fit_models
        }
for i in range(10):
    #pos_real += np.random.normal(0, 0.2, 2)
    pos_real += [0.1, 0]
    print(pos_real)
    measurement = measure(pos_real)
    for k, v in measurement.items():
        print(f"{v}\t{evaluate(test_env['parameters'][k], pos_real[0], pos_real[1])}\t{k}")
    print(score_particle(test_env, pos_real, measurement))
    motion_model(test_env, [0, 0])
    scores = score_env(test_env, measurement)
    for i in np.argsort(scores)[-10:]:
        print(test_env['pos'][i, :], scores[i])
    repopulate_particles(test_env, scores)
    pos_pred = np.mean(test_env['pos'], axis=0)
    print(max(scores), pos_pred)
    for k, v in measurement.items():
        print(f"{evaluate(test_env['parameters'][k], pos_pred[0], pos_pred[1])}\t{k}")
    plot_env(test_env, pos_real, pos_pred)
    input()
