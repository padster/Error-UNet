from __future__ import print_function, division

import numpy as np
import tensorflow as tf

HACK_USE_LAST_VALUE = False

class ErrorUNet:
    # Create network given a bunch of config parameters.
    def __init__(self, depth, topLayerWindow, kernelsPerLayer, learningRate, betweenLayerSize=1, errorWeight=1, batchSize=1):
        # variables:
        self.beVars = []
        self.eeVars = []
        self.feVars = []
        self.bfVars = []
        self.efVars = []
        self.ffVars = []

        # Placeholders:
        self.oldFLayers = []
        self.oldELayers = []
        self.ELayers = []
        self.FLayers = []

        # State:
        self.prevELayers = []

        # Optimization:
        self.errorLoss = None
        self.predictionLoss = None
        self.totalLoss = None

        self.depth = depth
        self.windowSize = topLayerWindow * (2 ** (depth - 1))
        self.kernelsPerLayer = kernelsPerLayer
        # Input fed into the network. (batchSize x windowSize)
        self.inputHolder = tf.placeholder(tf.float32, [batchSize, self.windowSize], name="In_F_0")
        # Output of the network. (batchSize x windowSize)
        self.outputHolder = tf.placeholder(tf.float32, [batchSize, self.windowSize], name="Out")

        # Create Variables:
        for at in range(depth):
            atS = str(at)
            self.eeVars.append(self.batch1dConv(               2, at == 0,   False, "eeVar_" + atS)) # mega hack: transposed conv2d invert out & in channel params
            self.feVars.append(self.batch1dConv(betweenLayerSize, at == 0, at == 0, "feVar_" + atS))
            self.efVars.append(self.batch1dConv(betweenLayerSize, at == 0, at == 0, "efVar_" + atS))
            self.ffVars.append(self.batch1dConv(               2, at == 1,   False, "ffVar_" + atS))

        # Old Forward value placeholders:
        for at in range(depth):
            atS = str(at)
            windowSize = topLayerWindow * (2 ** (depth - 1 - at))
            print ("NODES on level " + str(at) + " = " + str(windowSize))
            channels = 1 if at == 0 else self.kernelsPerLayer
            self.oldELayers.append(tf.placeholder(tf.float32, [batchSize, windowSize, channels], name="oldE_" + atS))
            self.prevELayers.append(np.random.randn(batchSize, windowSize, channels))
            self.beVars.append(self.batch1dConv(windowSize, at == 0, at == 0, "beVar_" + atS, isBias=True))
            self.bfVars.append(self.batch1dConv(windowSize, at == 0, at == 0, "bfVar_" + atS, isBias=True))

        print("BUILDING, old E -> old F")
        # New Forward value placeholders:
        lastLayer = None
        for at in range(depth):
            print ("Wiring OLD FORDWARD level " + str(at) + "...")
            fLayer = None
            if at == 0:
                # B x W => B x W x 1
                inputExpanded = tf.expand_dims(self.inputHolder, 2)
                fLayer = inputExpanded
            else:
                fLayer = self.convWithBias("oldF_" + str(at), [
                    (lastLayer, self.ffVars[at], 'FF'),
                    (self.oldELayers[at], self.efVars[at], 'EF')
                ], self.bfVars[at])
            self.oldFLayers.append(fLayer)
            lastLayer = fLayer


        # New Error value placeholders:
        print("BUILDING, old F -> new E")
        lastLayer = None
        for at in range(depth - 1, -1, -1):
            print ("Wiring ERROR level " + str(at) + "...")
            eLayer = None
            if at == depth - 1:
                eLayer = self.convWithBias("E_" + str(at), [
                    (self.oldFLayers[at], self.feVars[at], 'FE')
                ], self.beVars[at])
            else:
                eLayer = self.convWithBias("E_" + str(at), [
                    (lastLayer, self.eeVars[at], 'EE'),
                    (self.oldFLayers[at], self.feVars[at], 'FE')
                ], self.beVars[at])
            self.ELayers.append(eLayer)
            lastLayer = eLayer
        self.ELayers.reverse()

        # New Forward value placeholders:
        print("BUILDING, new E -> new F")
        lastLayer = None
        for at in range(depth):
            print ("Wiring FORDWARD level " + str(at) + "...")
            fLayer = None
            if at == 0:
                # B x W => B x W x 1
                inputExpanded = tf.expand_dims(self.inputHolder, 2)
                fLayer = inputExpanded
            else:
                fLayer = self.convWithBias("F_" + str(at), [
                    (lastLayer, self.ffVars[at], 'FF'),
                    (self.ELayers[at], self.efVars[at], 'EF')
                ], self.bfVars[at])
            self.FLayers.append(fLayer)
            lastLayer = fLayer

        # Debugging code:
        print ("E and F layer sizes:")
        for el in self.ELayers:
            print (el.get_shape())
        for fl in self.FLayers:
            print (fl.get_shape())

        # B x W => B x W x 1
        outHolderResized = tf.expand_dims(self.outputHolder, 2)

        if errorWeight == 0:
            # Without error loss, use the E layers as predictions (like UNet), not errors.
            self.prediction = self.ELayers[0]
            self.errorLoss = tf.constant(0)
            self.predictionLoss = tf.nn.l2_loss(self.prediction - outHolderResized) / self.windowSize
            self.totalLoss = self.predictionLoss
        else:
            # Output = Last Input plus predicted error
            self.prediction = self.FLayers[0] + self.ELayers[0]
            allNewEButFirst = tf.concat(self.ELayers[1:], 1)
            allOldFButFirst = tf.concat(self.oldFLayers[1:], 1)
            allNewFButFirst = tf.concat(self.FLayers[1:], 1)
            actualErrors = (allNewFButFirst - allOldFButFirst)
            self.errorLoss = (
                tf.nn.l2_loss(actualErrors - allNewEButFirst) +
                tf.nn.l2_loss((self.FLayers[0] - self.oldFLayers[0]) - self.ELayers[0]) +
                tf.nn.l2_loss(allNewEButFirst)
            ) / self.windowSize
            self.predictionLoss = tf.nn.l2_loss(self.prediction - outHolderResized) / self.windowSize
            self.totalLoss = tf.add(self.errorLoss * errorWeight, self.predictionLoss)

        globalStep = tf.Variable(0, trainable=False)
        decayedLearningRate = \
            tf.train.exponential_decay(learningRate, globalStep, 100, 0.75, staircase=True)
        self.optimizer = \
            tf.train.AdamOptimizer(decayedLearningRate).minimize(self.totalLoss)
            # tf.train.GradientDescentOptimizer(decayedLearningRate).minimize(self.totalLoss)

    def batch1dConv(self, width, firstLayer, lastLayer, name, isBias=False):
        inChannels = 1 if firstLayer else self.kernelsPerLayer
        outChannels = 1 if lastLayer else self.kernelsPerLayer
        if isBias:
            # bias = B x W x C
            filt = tf.Variable(np.random.randn(width, inChannels) * 0.1, dtype=tf.float32, name=name)
            return tf.expand_dims(filt, 0)
        else:
            # kernels = W x C x C
            return tf.Variable(np.random.randn(width, inChannels, outChannels), dtype=tf.float32, name=name)

    def convWithBias(self, name, convolutions, bias):
        # TODO: Dilation, not stride?
        result = bias
        print ("  * bias shape (= output shape): " + str(bias.get_shape()))
        for convolution in convolutions:
            value, kernel, cType = convolution
            print ("    + conv value shape:" + str(value.get_shape()))
            print ("    * conv kernel shape: " + str(kernel.get_shape()))
            print ("    * conv type: " + cType)
            convResult = None
            if cType == 'EE':
                convResult = self.eeConv(value, kernel, name)
            elif cType == 'FF':
                convResult = self.ffConv(value, kernel)
            else:
                assert cType == 'FE' or cType == 'EF'
                convResult = self.efeConv(value, kernel)
            print ("      = result: " + str(convResult.get_shape()))
            result = tf.add(result, convResult)
        activation = tf.nn.tanh(result)
        # activation = tf.nn.relu(result)
        # activation = tf.nn.tan(tf.nn.tanh(result))
        return tf.identity(activation, name=name + "_activation")

    def eeConv(self, eIn, kernel, name):
        # e layer 1D -> 2D: B x We x C -> B x We x 1 x C
        eIn2d = tf.expand_dims(eIn, 2)
        # filter, 1D -> 2D: Wk x C x C -> Wk x 1 x C x C
        kernel2d = tf.expand_dims(kernel, 1)
        eInShape = eIn2d.get_shape().as_list()
        kShape = kernel2d.get_shape().as_list()
        outShape = (eInShape[0], eInShape[1] * 2, eInShape[2], kShape[2])
        print ("EIn2d shape = " + str(eIn2d.get_shape()))
        print ("kernel2d shape = " + str(kernel2d.get_shape()))
        print ("OutShape = " + str(outShape))
        result = tf.nn.conv2d_transpose(eIn2d, kernel2d, output_shape=outShape, strides=[1, 2, 2, 1], name=name)
        return tf.squeeze(result, [2])

    def ffConv(self, fIn, kernel):
        return tf.nn.conv1d(fIn, kernel, stride=2, padding='VALID')

    def efeConv(self, value, kernel):
        return tf.nn.conv1d(value, kernel, stride=1, padding='SAME')

    def train(self, sess, inTensor, outTensor):
        # Given input and true output, update weights based from the optimizer
        if HACK_USE_LAST_VALUE:
            prediction = np.roll(inTensor, -1, axis=1)
            prediction[0, -1] = prediction[0, -2]
            err = outTensor - prediction
            errl2 = sum(sum(err ** 2)) / 2
            pLoss = errl2 / self.windowSize
            return 0, pLoss, pLoss

        inData = {
            self.inputHolder: inTensor,
            self.outputHolder: outTensor,
        }
        for i in range(len(self.oldFLayers)):
            inData[self.oldELayers[i]] = self.prevELayers[i]

        opt, eLayers, eLoss, pLoss, tLoss = sess.run(
            [self.optimizer, self.ELayers, self.errorLoss, self.predictionLoss, self.totalLoss],
            feed_dict=inData
        )
        self.prevELayers = eLayers
        return eLoss, pLoss, tLoss

    def generate(self, sess, inTensor):
        # Given input, generate the predicted output.
        if HACK_USE_LAST_VALUE:
            prediction = np.roll(inTensor, -1, axis=1)
            prediction[0, -1] = prediction[0, -2]
            return prediction

        inData = {
            self.inputHolder: inTensor,
        }
        for i in range(len(self.oldELayers)):
            inData[self.oldELayers[i]] = self.prevELayers[i]
        prediction, eLayers = sess.run([self.prediction, self.ELayers], feed_dict=inData)
        self.prefELayers = eLayers
        return prediction[:, :, 0]

    def debugValues(self, sess, inTensor):
        # Print all the variables and placeholders when given a particular input.
        inData = {
            self.inputHolder: inTensor,
        }
        for i in range(len(self.oldELayers)):
            inData[self.oldELayers[i]] = self.prevELayers[i]

        [beVars, eeVars, feVars, bfVars, efVars, ffVars, ELayers, FLayers] = sess.run(
            [self.beVars, self.eeVars, self.feVars, self.bfVars, self.efVars, self.ffVars, self.ELayers, self.FLayers],
            feed_dict=inData
        )
        self.debugBE(beVars)
        self.debugEE(eeVars)
        self.debugFE(feVars)
        self.debugBF(bfVars)
        self.debugEF(efVars)
        self.debugFF(ffVars)
        self.debugOldF(self.prevFLayers)
        self.debugE(ELayers)
        self.debugF(FLayers)

    def debugBE(self, beVars):
        for i in range(len(beVars)):
            print ("be_%d = %s" % (i, str(beVars[i][0, :, 0])))

    def debugEE(self, eeVars):
        for i in range(len(eeVars)):
            print ("ee_%d = %s" % (i, str(eeVars[i][0, :, 0])))

    def debugFE(self, feVars):
        for i in range(len(feVars)):
            print ("fe_%d = %s" % (i, str(feVars[i][0, :, 0])))

    def debugBF(self, bfVars):
        for i in range(len(bfVars)):
            print ("bf_%d = %s" % (i, str(bfVars[i][0, :, 0])))

    def debugEF(self, efVars):
        for i in range(len(efVars)):
            print ("ef_%d = %s" % (i, str(efVars[i][0, :, 0])))

    def debugFF(self, ffVars):
        for i in range(len(ffVars)):
            print ("ff_%d = %s" % (i, str(ffVars[i][0, :, 0])))

    def debugOldF(self, oldFLayers):
        for i in range(len(oldFLayers)):
            print ("oldF_%d = %s" % (i, str(oldFLayers[i][0, :, 0])))

    def debugE(self, ELayers):
        for i in range(len(ELayers)):
            print ("E_%d = %s" % (i, str(ELayers[i][0, :, 0])))

    def debugF(self, FLayers):
        for i in range(len(FLayers)):
            print ("F_%d = %s" % (i, str(FLayers[i][0, :, 0])))
