#include <string>
#include <vector>
#include <filesystem>

ROOT::RVecF take_lepton(const ROOT::RVecF &lept_var,const ROOT::RVecI &lept_idx) {
  ROOT::RVecF out;                                                                   
  out.reserve(lept_idx.size());
  
  for (int i : lept_idx) {
    if (i >= 0 && i < static_cast<int>(lept_var.size())) out.push_back(lept_var[i]);
    else out.push_back(-999.f);
  }
  return out;
}





void filteroots(std::string homepath) {
  std::vector<std::string> rootFiles;

  clog << "Entering filteroots func\n";
  for (const auto& entry : std::filesystem::directory_iterator(homepath)) {
    if (entry.is_regular_file() && entry.path().extension() == ".root") {
       rootFiles.push_back(entry.path().string());
    }
  }
  
  ROOT::RDataFrame fchain("Events", rootFiles);

    auto newfchain =  fchain.Define(
                                    "GoodFatJet_mask", "(FatJet_muonIdx3SJ >= 0) && (FatJet_electronIdx3SJ >= 0)")
                            .Filter("ROOT::VecOps::Any(GoodFatJet_mask)","Require at least one valid fat jet")
                            .Define("GoodMuonIdx", "FatJet_muonIdx3SJ[GoodFatJet_mask]")
                            .Define("GoodElectronIdx", "FatJet_electronIdx3SJ[GoodFatJet_mask]")
                            .Define("MuonJet_eta",take_lepton,{"Muon_eta","GoodMuonIdx"})
                            .Define("MuonJet_phi",take_lepton,{"Muon_phi","GoodMuonIdx"})
                            .Define("MuonJet_pt",take_lepton,{"Muon_pt","GoodMuonIdx"})
                            .Define("ElecJet_eta",take_lepton,{"Electron_eta","GoodElectronIdx"})
                            .Define("ElecJet_phi",take_lepton,{"Electron_phi","GoodElectronIdx"})
                            .Define("ElecJet_pt",take_lepton,{"Electron_pt","GoodElectronIdx"})
                            .Define("GoodFatJet_pt",  "FatJet_pt[GoodFatJet_mask]")
                            .Define("GoodFatJet_eta", "FatJet_eta[GoodFatJet_mask]")
                            .Define("GoodFatJet_phi", "FatJet_phi[GoodFatJet_mask]")
                            .Define("GoodFatJet_btagCSVV2", "FatJet_btagCSVV2[GoodFatJet_mask]")
                            .Define("GoodFatJet_btagDeepB", "FatJet_btagDeepB[GoodFatJet_mask]")
                            .Define("GoodFatJet_btagHbb", "FatJet_btagHbb[GoodFatJet_mask]");
//                            .Define("Good_SV_charge", "SV_charge")
//                            .Define("Good_SV_dlen", "SV_dlen")
//                            .Define("Good_SV_dxy", "SV_dxy[GoodFatJet_mask]")
//                            .Define("Good_SV_eta", "SV_eta[GoodFatJet_mask]")
//                            .Define("Good_SV_phi", "SV_phi[GoodFatJet_mask]")
//                            .Define("Good_SV_pt", "SV_pt[GoodFatJet_mask]")
//                            .Define("Good_SV_x", "SV_x[GoodFatJet_mask]")
//                            .Define("Good_SV_y", "SV_y[GoodFatJet_mask]")
//                            .Define("Good_SV_z", "SV_z[GoodFatJet_mask]")
//                            .Define("Good_SV_pAngle", "SV_pAngle[GoodFatJet_mask]")
//                            .Define("Good_SV_ntracks", "SV_ntracks[GoodFatJet_mask]");


//  "Muon_eta[FatJet_muonIdx3SJ]"
  newfchain.Snapshot("Events","dataset_SV_and.root",{"MuonJet_eta",
                        "MuonJet_phi",
                        "MuonJet_pt",
                        "ElecJet_eta",
                        "ElecJet_phi",
                        "ElecJet_pt",
                        "GoodFatJet_phi",
                        "GoodFatJet_eta",
                        "GoodFatJet_pt",
                        "GoodFatJet_btagCSVV2",
                        "GoodFatJet_btagDeepB",
                        "GoodFatJet_btagHbb",
                        "SV_charge", 
                        "SV_dlen",
                        "SV_dxy",
                        "SV_eta",
                        "SV_phi",
                        "SV_pt",
                        "SV_x",
                        "SV_y",
                        "SV_z",
                        "SV_pAngle", 
                        "SV_ntracks"
                        });
}



