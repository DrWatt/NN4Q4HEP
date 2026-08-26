import ROOT
import numpy as np
import csv
import os.path
import tensorflow as tf
import keras
import pennylane as pl
    
os.environ["KERAS_BACKEND"] = "tensorflow"

tf.get_logger().setLevel("ERROR")

file_in = ROOT.TFile("reduced.root")
bs = file_in.Get("Events")
#header = []
#print(bs)
branches = [b.GetName() for b in bs.GetListOfBranches()]
#header.insert(0,"Event")
#for b in branches: header.append(b)
#with open("reduced.csv", 'w') as file:
#    writer = csv.writer(file)
#    writer.writerow(header)
#
#    for entry in bs:
#        print(entry)
#        newb = [entry.event.eventNumber]
#        for b in branches:
#            newb.append(eval(branch))
#        print(newb)
df = ROOT.RDataFrame("Events","reduced_w_tags.root")
#npdf = df[0].AsNumpy(branches)
#print(npdf)
#np.save("reduced.npy", npdf)
dl = ROOT.Experimental.ML.RDataLoader(df, 128, 128, columns = ['MuonJet_eta', 'MuonJet_phi', 'MuonJet_pt', 'ElecJet_eta', 'ElecJet_phi', 'ElecJet_pt', 'GoodFatJet_phi', 'GoodFatJet_eta','GoodFatJet_pt'], target = ["GoodFatJet_btagCSVV2"], shuffle = True, drop_remainder = True,max_vec_sizes={'MuonJet_eta':1, 'MuonJet_phi':1, 'MuonJet_pt':1, 'ElecJet_eta':1, 'ElecJet_phi':1, 'ElecJet_pt':1, 'GoodFatJet_phi':1, 'GoodFatJet_eta':1,'GoodFatJet_pt':1,"GoodFatJet_btagCSVV2":1})

#print(X.as_numpy())









inp_col = dl.feature_columns
num_features = len(inp_col)

#help(dl)
#['MuonJet_eta', 'MuonJet_phi', 'MuonJet_pt', 'ElecJet_eta', 'ElecJet_phi', 'ElecJet_pt', 'GoodFatJet_phi', 'GoodFatJet_eta', 'GoodFatJet_pt']
#inputs = keras.Input(shape=(num_features,))
#
#x = keras.layers.Dense(64)(inputs)
#x = keras.layers.Dense(128, activation ='relu')(x)
#x = keras.layers.Dense(256, activation = 'relu')(x)
#x = keras.layers.Dense(64, activation = 'relu')(x)
#
#outs = keras.layers.Dense(3, activation = 'sigmoid')(x)
#
#model = keras.Model(inputs = inputs, outputs = outs)
#model.summary()
#model.compile(optimizer=keras.optimizers.Adam(learning_rate= 1e-3), loss= "mse", metrics=['accuracy'])
#epochs = 1
#dl_repeated = dl.as_tensorflow().repeat(epochs)
#num_batches = dl.num_batches
#
#
#model.fit(dl_repeated,steps_per_epoch=num_batches,epochs=epochs)
dev = pl.device('default.qubit', wires = 9)
shape = pl.StronglyEntanglingLayers.shape(n_layers=2, n_wires=len(dev.wires))
print(shape)
wei = tf.Variable(
    tf.random.uniform(
        shape,
        minval=-np.pi/2,
        maxval=np.pi/2,
        dtype=tf.float64
    ),
    trainable=True
)
obs = pl.PauliZ(0)
#for i in range(1,9):
#    obs = obs @ pl.PauliZ(i)
@pl.batch_input(argnum=0)
@pl.qnode(dev, interface='tf')
def circuit(dl, wei):
    pl.AngleEmbedding(dl, wires = dev.wires, rotation ="Y")
    #for i in dev.wires:
    #    pl.RY(dl[i], wires = i)
    #w = tf.reshape(wei, (1,9))
    #pl.BasicEntanglerLayers(wei,dev.wires)
    pl.StronglyEntanglingLayers(weights=wei, wires=dev.wires)
    #qp.PhaseShift(theta, wires=0)
    
    return pl.expval(obs)




opt = keras.optimizers.Adam(learning_rate = 1e-1)

trainset = dl.as_tensorflow()
dummy = tf.Variable(tf.random.uniform(shape=(1,9,),minval=-0.1,maxval=0.1,dtype=tf.float64))

print(dl.num_batches)
#print(pl.draw(circuit, level = "device")(dummy,wei))
for step, (X,Y) in enumerate(trainset):
    if step == 0:
        print(pl.draw(circuit, level = "device")([X[0]],wei))
    X = tf.cast(X, tf.float64)
    Y = tf.cast(Y, tf.float64)
    #print(f"\rStep: {step} ",end="", flush=True)
    target = tf.squeeze(Y, axis=-1)
    with tf.GradientTape() as tape:
        pred = (circuit(X, wei) + 1 )/2
        loss = tf.reduce_mean(tf.math.square(target - pred))
    gradients = tape.gradient(loss, wei)
    opt.apply_gradients([(gradients, wei)])
    tf.print(
        "Step:", step,
        "Loss:", loss,
        "Truth:", target[:5],
        "Pred:", pred[:5],
        #"Grad:", gradients[:5]
    )

