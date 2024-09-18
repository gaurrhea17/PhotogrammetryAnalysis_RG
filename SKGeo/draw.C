void draw(float Input_Phi=-1) {

#include "../pars.h";
  
  gStyle->SetTitleOffset(1.3,"y"); 
  gStyle->SetLabelSize(0.04,"x");
  gStyle->SetTitleSize(0.05,"x");
  gStyle->SetLabelSize(0.04,"y");
  gStyle->SetTitleSize(0.05,"y");
  gStyle->SetLabelSize(0.04,"z");
  gStyle->SetTitleSize(0.05,"z");
  
  TFile *infile = new TFile("../DarkRates/ConnectionTable_SK5.root");
  TTree *ConnectionTable = (TTree*)infile->Get("ConnectionTable");

  // Rotate to view normal to light injectors
  float PhiOfInjectors = atan(InjectorPositions[0][0][1]/InjectorPositions[0][0][0]);
  cout << "Angle of Injectors = " << PhiOfInjectors << " " << PhiOfInjectors * 180 / TMath::Pi() << endl;
  TString AxisTitle[2] = {
    "#perp  to Injector Normal (cm)",
    "#parallel  to Injector Normal (cm)"
  };
  
  float RotationAngle = -(TMath::Pi()/2 - PhiOfInjectors);

  if (Input_Phi>=0) {
    RotationAngle = -(90 - Input_Phi) * TMath::Pi() / 180;
    AxisTitle[0] = Form("#perp to #phi = %.1f#circ (cm)", Input_Phi);
    AxisTitle[1] = Form("#parallel to #phi = %.1f#circ (cm)", Input_Phi);
  }  

  // Big zoom and label PMTs
   Int_t           cableid;
   Int_t           supserial;
   Int_t           modserial;
   Int_t           hutnum;
   Int_t           tkobnum;
   Int_t           tkomodadd;
   Int_t           qbch;
   Int_t           hvcrate;
   Int_t           hvmodadd;
   Int_t           hvch;
   Float_t         oldhv;
   Int_t           pmtflag;
   Int_t           pmtx;
   Int_t           pmty;
   Int_t           pmtz;
   Float_t         prodyear_sk4;
   Float_t         prodyear_sk5;

   TTree *fChain = ConnectionTable;
   fChain->SetBranchAddress("cableid", &cableid);
   fChain->SetBranchAddress("supserial", &supserial);
   fChain->SetBranchAddress("modserial", &modserial);
   fChain->SetBranchAddress("hutnum", &hutnum);
   fChain->SetBranchAddress("tkobnum", &tkobnum);
   fChain->SetBranchAddress("tkomodadd", &tkomodadd);
   fChain->SetBranchAddress("qbch", &qbch);
   fChain->SetBranchAddress("hvcrate", &hvcrate);
   fChain->SetBranchAddress("hvmodadd", &hvmodadd);
   fChain->SetBranchAddress("hvch", &hvch);
   fChain->SetBranchAddress("oldhv", &oldhv);
   fChain->SetBranchAddress("pmtflag", &pmtflag);
   fChain->SetBranchAddress("pmtx", &pmtx);
   fChain->SetBranchAddress("pmty", &pmty);
   fChain->SetBranchAddress("pmtz", &pmtz);
   fChain->SetBranchAddress("prodyear_sk4", &prodyear_sk4);
   fChain->SetBranchAddress("prodyear_sk5", &prodyear_sk5);

   const int nSurfaces = 3;
   enum surface_enum {top, barrel, bottom};
   TString SurfaceNames[nSurfaces] = {"Top Endcap", "Barrel", "Bottom Endcap"};
   
   TGraph *gr_PMTs[nSurfaces];
   for (int isurface=0; isurface<nSurfaces; isurface++) 
     gr_PMTs[isurface] = new TGraph();
   
   int nPoints[nSurfaces] = {0};
   
   for (int ientry=0; ientry<fChain->GetEntries(); ientry++) {
     fChain->GetEntry(ientry);

     float RotatedPosition[2] = { pmtx*cos(RotationAngle) + pmty*sin(RotationAngle),
				  -pmtx*sin(RotationAngle) + pmty*cos(RotationAngle) };
 
     //if (RotatedPosition[0] < ZoomRange[0][0] || RotatedPosition[0] > ZoomRange[0][1] ||
     //pmtz < ZoomRange[1][0] || pmtz > ZoomRange[1][1] )       continue;
     
     //if (pmtflag!=6) continue;
     int isurface = -1;

     if (pmtz > 1800) {
       isurface = top;
       gr_PMTs[isurface]->SetPoint(nPoints[isurface], RotatedPosition[0], RotatedPosition[1]);
     }

     else if (pmtz < -1800) {
       isurface = bottom;
       gr_PMTs[isurface]->SetPoint(nPoints[isurface], RotatedPosition[0], RotatedPosition[1]);
     }       

     else {
       if (RotatedPosition[1] < 0) continue;  // Only facing wall
       isurface = barrel;
       gr_PMTs[isurface]->SetPoint(nPoints[isurface], RotatedPosition[0], (float)pmtz);
     }
     nPoints[isurface]++;
     

   }

   for (int isurface=0; isurface<nSurfaces; isurface++) {
     
     TCanvas *c_surface = new TCanvas(Form("c_surface_%d", isurface), Form("c_surface_%d", isurface), 10, 10, 900, 1000);
     
     // Draw PMT points
     gr_PMTs[isurface]->SetTitle(SurfaceNames[isurface]);

     if (isurface == barrel) {
       gr_PMTs[isurface]->GetXaxis()->SetTitle(AxisTitle[0]);
       gr_PMTs[isurface]->GetYaxis()->SetTitle("z (cm)");
     }
     else {
       gr_PMTs[isurface]->GetXaxis()->SetTitle(AxisTitle[0]);
       gr_PMTs[isurface]->GetYaxis()->SetTitle(AxisTitle[1]);
     }
     
     gr_PMTs[isurface]->Draw("AP");
     

     // Draw light injectors
     TMarker *Injector;
     for (int iSystem=0; iSystem<nSystems; iSystem++) {
       for (int iInjector=0; iInjector<nInjectors; iInjector++) {
	 
	 float InjectorPositionRotated[2] = {
	   InjectorPositions[iSystem][iInjector][0]*cos(RotationAngle) + InjectorPositions[iSystem][iInjector][1]*sin(RotationAngle),
	   -InjectorPositions[iSystem][iInjector][0]*sin(RotationAngle) + InjectorPositions[iSystem][iInjector][1]*cos(RotationAngle)
	 };

	 if (isurface == barrel) {
	   if (InjectorPositionRotated[1] < 0) continue;
	   Injector = new TMarker(InjectorPositionRotated[0], InjectorPositions[iSystem][iInjector][2], 1);
	 }
	 
	 else 
	   Injector = new TMarker(InjectorPositionRotated[0], InjectorPositionRotated[1], 1);
	 
	 Injector->SetMarkerStyle(34-iSystem);
	 Injector->SetMarkerSize(2);
	 Injector->SetMarkerColor(kBlue-iSystem*7);
	 Injector->Draw("same");
       }
     }
     
     
     // Draw PMT labels
     for (int ientry=0; ientry<fChain->GetEntries(); ientry++) {
       fChain->GetEntry(ientry);

       TString PMT_PrintVar = Form("%d", cableid);
       //TString PMT_PrintVar = Form("%d", pmtflag);
       //     int SuperModuleScale = 211-supserial;
       
       float RotatedPosition[2] = { pmtx*cos(RotationAngle) + pmty*sin(RotationAngle),
				    -pmtx*sin(RotationAngle) + pmty*cos(RotationAngle) };
       
       TText *PMTLabel = new TText(); 
       PMTLabel->SetTextSize(0.015);

       PMTLabel->SetTextColor(supserial%7+2);
       if (pmtflag == 6) PMTLabel->SetTextColor(kBlack);

       if (isurface == barrel) {
	 if (fabs(pmtz) > 1800) continue;
	 if (RotatedPosition[1] < 0) continue;
	 PMTLabel->DrawText(RotatedPosition[0], pmtz, PMT_PrintVar);
       }

       else if (isurface == top) {
	 if (pmtz > 1800) 
	   PMTLabel->DrawText(RotatedPosition[0], RotatedPosition[1], PMT_PrintVar);
       }

       else { // Bottom
	 if (pmtz < -1800) 
	   PMTLabel->DrawText(RotatedPosition[0], RotatedPosition[1], PMT_PrintVar);
       }
     }
   }

   
   // Check rotation math
   //c_top->cd();
   //int iSystem =0, iInjector = 0;
   //float InjectorPositionRotated[2] = {
   //  InjectorPositions[iSystem][iInjector][0]*cos(RotationAngle) + InjectorPositions[iSystem][iInjector][1]*sin(RotationAngle),
   //  -InjectorPositions[iSystem][iInjector][0]*sin(RotationAngle) + InjectorPositions[iSystem][iInjector][1]*cos(RotationAngle)
   //};
   //TMarker *Injector_TopViewRotated = new TMarker(InjectorPositionRotated[0], InjectorPositionRotated[1], 1);
   //Injector_TopViewRotated->SetMarkerStyle(33);
   //Injector_TopViewRotated->SetMarkerSize(2);
   //Injector_TopViewRotated->SetMarkerColor(kBlue-7);
   //Injector_TopViewRotated->Draw("same");

}
