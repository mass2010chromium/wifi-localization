import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#datafile = "94:64:24:99:d7:61-eduroam.data"
#datafile = "94:64:24:99:d7:62-IllinoisNet_Guest.data"
#datafile = "94:64:24:9b:46:50-IllinoisNet.data"
#datafile = "94:64:24:9b:b7:30-IllinoisNet.data"
# 94:64:24:9b:b7:31-eduroam.data
# 94:64:24:9b:b7:32-IllinoisNet_Guest.data
#datafile = "94:64:24:9d:2e:01-eduroam.data"
#datafile = "94:64:24:9d:2e:02-IllinoisNet_Guest.data"

data = np.load(datafile, allow_pickle=True)
import scipy.interpolate

#01:00:E4:C3:2A:76-Ronin.data

raw_xs = data[:, 0]
raw_ys = data[:, 1]
truth = data[:, 2]

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
sf = None

def plot(plot_data, ax=ax, **kwargs):
    global sf
    model = scipy.interpolate.LinearNDInterpolator(plot_data[:, :2], plot_data[:, 2])
    xs = np.linspace(np.min(plot_data[:,0]), np.max(plot_data[:,0]))
    ys = np.linspace(np.min(plot_data[:,1]), np.max(plot_data[:,1]))
    X, Y = np.meshgrid(xs, ys)
    _X, _Y = X.reshape(-1), Y.reshape(-1)
    C = model(_X, _Y).reshape(X.shape)
    min_color = np.nanmin(plot_data[:, 2])
    max_color = np.nanmax(plot_data[:, 2])
    if 'sf' in kwargs:
        sf = kwargs['sf']
    elif sf is not None:
        sf.remove()
    sf = ax.plot_surface(X, Y, C, vmin=min_color, vmax=max_color, cmap=cm.coolwarm)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

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

#params = [0.8, 0.8, 0.0001]
params = [0, 0, max(truth)]
#params = [0.4, 0.4, 0.0001]

lr = 1
niter = 100

def fit(xs, ys, zs, niter):
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    A0 = max(zs)

    x = np.random.uniform(min_x, max_x, (niter, 1))
    y = np.random.uniform(min_y, max_y, (niter, 1))
    A = np.ones((niter, 1)) * A0
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

def fit2(xs, ys, zs, niter):
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    A0 = max(zs)

    x = np.random.uniform(min_x, max_x, (niter, 2, 1))
    y = np.random.uniform(min_y, max_y, (niter, 2, 1))
    A = np.ones((niter, 2, 1)) * A0
    for i in range(5000):
        zvals = evaluate((x, y, A), xs, ys)
        totals = np.sum(zvals, axis=1)
        err_deriv = (1000*(totals - zs)*2 / len(zs)).reshape(niter, 1, -1)
        x_grad, y_grad, A_grad = grad((x, y, A), xs, ys)
        x -= lr * np.sum(x_grad * err_deriv, axis=2).reshape((-1, 2, 1))
        y -= lr * np.sum(y_grad * err_deriv, axis=2).reshape((-1, 2, 1))
        A -= lr * np.sum(A_grad * err_deriv, axis=2).reshape((-1, 2, 1))

    zvals = evaluate((x, y, A), xs, ys)
    totals = np.sum(zvals, axis=1)
    err = (1000*(totals - zs))**2 / len(zs)
    err_val = np.sum(err, axis=1)
    best = np.argmin(err_val)
    return err_val[best], (x[best, :, 0].reshape(-1, 1),
                       y[best, :, 0].reshape(-1, 1),
                       A[best, :, 0].reshape(-1, 1))

#err, params = fit(raw_xs, raw_ys, truth, niter)
#best_err, test_params = fit2(raw_xs, raw_ys, truth, niter)
#print(err, best_err)
#print(test_params)
#zvals = evaluate(test_params, raw_xs, raw_ys)
err, params = fit(raw_xs, raw_ys, truth, niter)
zvals = evaluate(params, raw_xs, raw_ys)
totals = zvals#np.sum(zvals, axis=0)
plot(np.array(list(zip(raw_xs, raw_ys, totals))))
fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
plot(data, ax2, sf=None)

plt.show()
