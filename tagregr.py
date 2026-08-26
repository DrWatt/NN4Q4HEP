import os
import ROOT
import numpy as np
import tensorflow as tf
import pennylane as pl
    
os.environ["KERAS_BACKEND"] = "tensorflow"

tf.get_logger().setLevel("ERROR")

def tree_loader(path: str):
    df = ROOT.RDataFrame("Events", path)
    
    dl = ROOT.Experimental.ML.RDataLoader(df, 128, 128, columns = ['MuonJet_eta', 'MuonJet_phi', 'MuonJet_pt', 'ElecJet_eta', 'ElecJet_phi', 'ElecJet_pt', 'GoodFatJet_phi', 'GoodFatJet_eta','GoodFatJet_pt'], target = ["GoodFatJet_btagCSVV2"], shuffle = True, drop_remainder = True,max_vec_sizes={'MuonJet_eta':1, 'MuonJet_phi':1, 'MuonJet_pt':1, 'ElecJet_eta':1, 'ElecJet_phi':1, 'ElecJet_pt':1, 'GoodFatJet_phi':1, 'GoodFatJet_eta':1,'GoodFatJet_pt':1,"GoodFatJet_btagCSVV2":1})
    
    inp_col = dl.feature_columns
    num_features = len(inp_col)
    print(f"Loaded tree file with {num_features} features.")
    return dl


dev = pl.device('default.qubit', wires = 9)
shape = pl.StronglyEntanglingLayers.shape(n_layers=2, n_wires=len(dev.wires))
obs = pl.PauliZ(0)
#for i in range(1,9):
#    obs = obs @ pl.PauliZ(i)

@pl.batch_input(argnum=0)
@pl.qnode(dev, interface='tf')
def circuit(dl, wei):
    pl.AngleEmbedding(dl, wires = dev.wires, rotation ="Y")
    #pl.BasicEntanglerLayers(wei,dev.wires)
    pl.StronglyEntanglingLayers(weights=wei, wires=dev.wires)
    
    return pl.expval(obs)


def train_epoch(trainset, weights, circuit, opt):
    for step, (X,Y) in enumerate(trainset):
        if step == 0:
            print(pl.draw(circuit, level = "device")([X[0]],weights))
        X = tf.cast(X, tf.float64)
        Y = tf.cast(Y, tf.float64)
        #print(f"\rStep: {step} ",end="", flush=True)
        target = tf.squeeze(Y, axis=-1)
        with tf.GradientTape() as tape:
            pred = (circuit(X, weights) + 1 )/2
            loss = tf.reduce_mean(tf.math.square(target - pred))
        gradients = tape.gradient(loss, weights)
        opt.apply_gradients([(gradients, weights)])
        tf.print(
            "Step:", step,
            "Loss:", loss,
            "Truth:", target[:5],
            "Pred:", pred[:5],
            #"Grad:", gradients[:5]
        )


if __name__ == "__main__":
    dataframe = tree_loader("reduced_w_tags.root")

    trainset = dataframe.as_tensorflow()

    print(f"The training set contains {dataframe.num_batches} batches of data")
    
    opt = tf.keras.optimizers.Adam(learning_rate = 1e-1)
    weights = tf.Variable(tf.random.uniform(shape, minval=-np.pi/2, maxval=np.pi/2, dtype=tf.float64), trainable=True)
    train_epoch(trainset, weights, circuit, opt)
