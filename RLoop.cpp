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

    auto newfchain =  fchain.Define("GoodFatJet_mask", "(FatJet_muonIdx3SJ >= 0) || (FatJet_electronIdx3SJ >= 0)").Define("MuonJet_eta",take_lepton,{"Muon_eta","FatJet_muonIdx3SJ"}).Define("MuonJet_phi",take_lepton,{"Muon_phi","FatJet_muonIdx3SJ"}).Define("MuonJet_pt",take_lepton,{"Muon_pt","FatJet_muonIdx3SJ"}).Define("ElecJet_eta",take_lepton,{"Electron_eta","FatJet_electronIdx3SJ"}).Define("ElecJet_phi",take_lepton,{"Electron_phi","FatJet_electronIdx3SJ"}).Define("ElecJet_pt",take_lepton,{"Electron_pt","FatJet_electronIdx3SJ"}).Define("GoodFatJet_pt",  "FatJet_pt[GoodFatJet_mask]").Define("GoodFatJet_eta", "FatJet_eta[GoodFatJet_mask]").Define("GoodFatJet_phi", "FatJet_phi[GoodFatJet_mask]").Define("GoodFatJet_btagCSVV2", "FatJet_btagCSVV2[GoodFatJet_mask]").Define("GoodFatJet_btagDeepB", "FatJet_btagDeepB[GoodFatJet_mask]").Define("GoodFatJet_btagHbb", "FatJet_btagHbb[GoodFatJet_mask]");
//  "Muon_eta[FatJet_muonIdx3SJ]"
  newfchain.Snapshot("Events","reduced_w_tags.root",{"MuonJet_eta","MuonJet_phi","MuonJet_pt","ElecJet_eta","ElecJet_phi","ElecJet_pt","GoodFatJet_phi","GoodFatJet_eta","GoodFatJet_pt","GoodFatJet_btagCSVV2","GoodFatJet_btagDeepB","GoodFatJet_btagHbb"});
  //fchain.Snapshot("Events","reduced.root",{"FatJet_phi","FatJet_eta","FatJet_pt","Jet_nMuons",}); 
}



