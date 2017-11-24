import numpy as np

# [[ 704.54125977  705.24932861  705.95739746  706.66546631]] ==> [[ 705.21264648  704.37823486  706.07226562  707.37353516]]
be_0 = np.array([-0.17143546,  0.04736949,  0.05420579,  0.01406611])
be_1 = np.array([ 0.16508758, -0.2130283 ])
be_2 = np.array([-0.03586718])
ee_0 = np.array([ 1.16824698])
ee_1 = np.array([ 0.53905809])
ee_2 = np.array([ 0.52720374])
fe_0 = np.array([ 1.26014256])
fe_1 = np.array([-0.63585436])
fe_2 = np.array([-0.03894047])
bf_0 = np.array([ 0.00247608, -0.23149142, -0.02517184,  0.0283164 ])
bf_1 = np.array([ 0.0350868,  -0.10333713])
bf_2 = np.array([-0.04956153])
ef_0 = np.array([-0.06146076])
ef_1 = np.array([-0.36896238])
ef_2 = np.array([-0.53581744])
ff_0 = np.array([ 0.35132337])
ff_1 = np.array([ 0.06114592])
ff_2 = np.array([-0.11515216])
oldF_0 = np.array([ 0.2972393,  -0.79362488,  0.4937455,   0.40891872])
oldF_1 = np.array([-0.66907608,  0.46401612])
oldF_2 = np.array([-0.39499556])
E_0 = np.array([ 0.6714012,  -0.87111735,  0.11489763,  0.70807827])
E_1 = np.array([ 0.52228814, -0.48019657])
E_2 = np.array([-0.020483])
F_0 = np.array([ 705.24932861,  705.95739746,  706.66546631,  707.37353516])
F_1 = np.array([-1. -1.])
F_2 = np.array([ 0.27856579])

def tConv(t1, t2):
    print ("TCONV")
    print (t1)
    print (t2)
    v = []
    for v2 in t2:
        for v1 in t1:
            v.append(v1 * v2)
            v.append(v1 * v2)
    print (np.array(v))
    return np.array(v)

print ("E2 = tanh(oldF2 conv fe2 + be_2)")
print (E_2)
print (np.tanh(np.convolve(oldF_2, fe_2) + be_2))

print ("E1 = tanh(oldF1 conv fe1 + E2 tconv ee1 + be_1)")
print (E_1)
print (np.tanh(np.convolve(oldF_1, fe_1) + tConv(E_2, ee_1) + be_1))
# [[ 705.24932861  705.95739746  706.66546631  707.37353516]] ==> [[ 705.92071533  705.08630371  706.78033447  708.081604  ]]
