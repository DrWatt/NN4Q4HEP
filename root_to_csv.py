import ROOT
import numpy as np
import csv
import os.path
import tensorflow as tf
    
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
dl = ROOT.Experimental.ML.RDataLoader(df, 128, 50, columns = ['MuonJet_eta', 'MuonJet_phi', 'MuonJet_pt', 'ElecJet_eta', 'ElecJet_phi', 'ElecJet_pt', 'GoodFatJet_phi', 'GoodFatJet_eta','GoodFatJet_pt'], target = ["GoodFatJet_btagCSVV2","GoodFatJet_btagDeepB","GoodFatJet_btagHbb"], shuffle = True, drop_remainder = True,max_vec_sizes={'MuonJet_eta':1, 'MuonJet_phi':1, 'MuonJet_pt':1, 'ElecJet_eta':1, 'ElecJet_phi':1, 'ElecJet_pt':1, 'GoodFatJet_phi':1, 'GoodFatJet_eta':1,'GoodFatJet_pt':1,"GoodFatJet_btagCSVV2":1,"GoodFatJet_btagDeepB":1,"GoodFatJet_btagHbb":1})
X = dl.as_tensorflow()

#print(X.as_numpy())
#print(dl.num_batches)
for X,Y in dl.as_tensorflow():
    a = X
    b = Y
    break

print(a)
print("LABELS")
print(b)
#help(dl)
#['MuonJet_eta', 'MuonJet_phi', 'MuonJet_pt', 'ElecJet_eta', 'ElecJet_phi', 'ElecJet_pt', 'GoodFatJet_phi', 'GoodFatJet_eta', 'GoodFatJet_pt']
