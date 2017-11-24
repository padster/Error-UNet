from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

import inputReader
import errorUnet

def fullScreen():
    pass
    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())

def runTraining(net, data, epochs, runGeneration=True):
    usePredictionToGenerate = True

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)

        eLosses, pLosses, tLosses = [], [], []
        avELosses, avPLosses, avTLosses = [], [], []
        for epoch in tqdm(range(epochs)):
            # print ("Starting epoch %d:" % (epoch))
            for i in tqdm(range(len(data) - net.windowSize - 1)):
                prevInput = np.array([data[i + 0 : i + 0 + net.windowSize]])
                nextInput = np.array([data[i + 1 : i + 1 + net.windowSize]])
                eLoss, pLoss, tLoss = net.train(sess, prevInput, nextInput)
                eLosses.append(eLoss)
                pLosses.append(pLoss)
                tLosses.append(tLoss)
            avELosses.append(np.mean(eLosses))
            avPLosses.append(np.mean(pLosses))
            avTLosses.append(np.mean(tLosses))
            if epoch == epochs - 1:
                # Visualize results after the final pass:
                fullScreen()
                plt.plot(pLosses, label="P loss, final epoch")
                plt.plot(eLosses, label="E loss, final epoch")
                plt.plot(tLosses, label="Total loss, final epoch")
                plt.title('Final losses')
                plt.xlabel('Sample')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()
                fullScreen()
                plt.plot(avPLosses, label="Average Prediction loss")
                plt.plot(avELosses, label="Average Error loss")
                plt.plot(avTLosses, label="Average Total loss")
                plt.title('Average losses across epoch')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()
                print ("Average loss, final pass: Error = %.2e, Prediction = %.2e, Total = %.2e" % (
                    avELosses[-1], avPLosses[-1], avTLosses[-1]
                ))
                print ()
            eLosses, pLosses, tLosses = [], [], []

        eLosses, pLosses, tLosses = [], [], []
        predicted = data[:2 * net.windowSize].tolist()
        actual = data[:2 * net.windowSize].tolist()
        if runGeneration:
            # Pre-seed
            for i in range(net.windowSize):
                prevInput = np.array([data[i + 0 : i + 0 + net.windowSize]])
                nextInput = np.array([data[i + 1 : i + 1 + net.windowSize]])
                net.train(sess, prevInput, nextInput)

            # Generate
            inData = None
            for i in range(net.windowSize, len(data) - net.windowSize - 1):
                if inData is None:
                    inData = np.array([data[i : i + net.windowSize]])

                prediction = net.generate(sess, inData)
                nextSample = prediction[0][-1]
                predicted.append(nextSample)
                actual.append(data[i + net.windowSize])
                inData = np.roll(inData, -1, axis=1)
                inData[0][-1] = nextSample if usePredictionToGenerate else data[i + net.windowSize]
            # Show generated.
            fullScreen()
            plt.plot(predicted, label="Prediction")
            plt.plot(actual, label="Actual")
            plt.title('Prediction vs Actual')
            plt.legend()
            plt.show()

# Example training against straight line input
def testRunLine():
    depth = 4
    topWindowSize = 1
    kernelsPerLayer = 2
    learningRate = 0.001
    betweenLayerWidth = 3
    errorWeight = 1
    epochs = 15
    samples = 1000
    net = errorUnet.ErrorUNet(depth, topWindowSize, kernelsPerLayer, learningRate, betweenLayerWidth, errorWeight)
    data = inputReader.generateStraightLine(samples)
    runTraining(net, data, epochs)

# Example training against chaotic input
def testRunLorenz():
    depth = 5#4
    topWindowSize = 1#2
    kernelsPerLayer = 1#2
    learningRate = 0.001
    betweenLayerWidth = 1
    errorWeight = 1
    epochs = 30
    samples = 4000
    net = errorUnet.ErrorUNet(depth, topWindowSize, kernelsPerLayer, learningRate, betweenLayerWidth, errorWeight)
    data = inputReader.generateLorenzData(samples)
    runTraining(net, data, epochs)

# Example training against 'real' audio wave input
def testRunMusic():
    depth = 4
    topWindowSize = 1
    kernelsPerLayer = 2
    learningRate = 0.001
    betweenLayerWidth = 3
    errorWeight = 1
    epochs = 15
    samples = 2000
    net = errorUnet.ErrorUNet(depth, topWindowSize, kernelsPerLayer, learningRate, betweenLayerWidth, errorWeight)
    hz, data = inputReader.loadAllAudio(samples)
    print ("%f seconds of data" % (samples / hz))
    runTraining(net, data[0], epochs)


def main():
    # testRunLine()
    testRunLorenz()
    # testRunMusic()


if __name__ == '__main__': main()
