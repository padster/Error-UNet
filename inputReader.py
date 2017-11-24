from __future__ import print_function, division

import numpy as np
import scipy.io.wavfile as wavfile
from tqdm import tqdm

import matplotlib.pyplot as plt

SCALE = 1

# Read a .wav file, return sample rate and numpy array, normalized to [-1, 1]
def audioToNp(path):
    print ("Reading Audio from %s..." % path)
    rate, samples = wavfile.read(path)
    samples = samples[:, 0]
    return rate, samples

# Save a numpy array of samples back out to a wav file.
def npToAudio(path, rate, samples):
    print ("Writing Audio to %s..." % path)
    samples = samples * 32768.0
    samples = samples.astype(np.int16)
    wavfile.write(path, rate, samples)

# Loads all Audio files (same rate) into [Files x Samples] matrix
def loadAllAudio(samples):
    files = [
        "data/justAnotherDream.wav"
    ]
    result = np.zeros((len(files), samples))
    sharedHz = None
    for i in range(len(files)):
        hz, fullResult = audioToNp(files[i])
        start = len(fullResult) // 2
        result[i] = fullResult[start:start + samples] * SCALE
        if sharedHz is None:
            sharedHz = hz
        else:
            assert sharedHz == hz
    return sharedHz, result


def generateSinusoid(samples):
    # return np.arange(0, 20 * 2 * np.pi, 0.1) * 10
    return np.sin(np.arange(0, (samples // 314) * 2 * np.pi, 0.02)) * 10

def generateFromNetwork(samples, sess, net):
    print ("Generating samples from network...")
    result = np.zeros(samples)
    inData = np.random.randn(1, net.windowSize)
    for i in tqdm(range(samples)):
        if i == samples - 1:
            net.debugValues(sess, inData)
        prediction = net.generate(sess, inData)
        if i == samples - 1 or True:
            print (inData, end="")
            print (" ==> ", end="")
            print (prediction)
        inData = np.roll(inData, -1, axis=1)
        inData[0][-1] = prediction[0][-1]
        result[i] = prediction[0][-1]
    return result

def lorenz(x, y, z, s=10, r=28, b=2.667):
    '''
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    '''
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot

def generateLorenzData(samples):
    dt = 0.01
    xs = np.empty((samples + 1,))
    xs[0], y, z = 0., 1., 1.05
    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(samples):
        x_dot, y_dot, z_dot = lorenz(xs[i], y, z)
        xs[i + 1] = xs[i] + (x_dot * dt)
        y += y_dot * dt
        z += z_dot * dt
    xs = xs[1:]
    xs = (xs - np.min(xs)) / (np.max(xs) - np.min(xs))
    xs = 2.0 * xs - 1.0
    return xs

def generateStraightLine(samples):
    m = -1.4
    c = 0.8
    result = (m * np.arange(0.0, 1.0, 1.0 / (samples + 3)) + c)[:samples]
    return result
