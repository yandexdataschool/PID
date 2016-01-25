
#include "ChargedProtoANNPIDTeacher/rootstyle.h"

void setStyle()
{

  // reset style
  gROOT->SetStyle("Plain");

  // new style to setup
  TStyle * lhcbStyle= new TStyle("lhcbStyle","LHCb official plots style");

  // Define various values
  // ----------------------------------------------------------
  // use helvetica-bold-r-normal, precision 2 (rotatable)
  Int_t lhcbFont = 62;
  // Foreground colour
  Int_t lhcbForeColour = kBlack;
  // Colour of the general background
  Int_t lhcbPadColour = 10; //kWhite;
  // Colour of the canvas
  Int_t lhcbCanvasColour = 10; //kWhite;
  // Colour of the title box background
  //Int_t lhcbTitleBackColour = kWhite;
  // Colour of the stats box
  Int_t lhcbStatsColour = 10; //kWhite;
  // Histogram fill colour
  //Int_t lhcbFillColour = kWhite;
  // line thickness
  Int_t lhcbWidth = 1;
  // text font size
  Double_t lhcbTextSize = 0.03;
  // label text size
  Double_t lhcbLabelSize = 0.03;
  // Axis label text size
  Double_t lhcbAxisLabelSize = 0.03;
  // Axis label offset
  Double_t lhcbAxisLabelOffset = 1.4;
  // title text size
  //Double_t lhcbTitleSize = 0.06;
  // Marker type
  Int_t lhcbMarkerType = 20;
  // marker size
  Double_t lhcbMarkerSize = 0.8;
  // font size in statistics box
  Double_t lhcbStatFontSize = 0.035;
  // Width and height of stats box
  Double_t lhcbStatBoxWidth  = 0.2;
  Double_t lhcbStatBoxHeight = 0.15;

  // use plain black on white colors
  lhcbStyle->SetFrameBorderMode(0);
  lhcbStyle->SetPadBorderMode(0);
  lhcbStyle->SetPadColor(lhcbPadColour);
  lhcbStyle->SetStatColor(lhcbStatsColour);
  lhcbStyle->SetPalette(lhcbForeColour);
  ////lhcbStyle->SetFillColor(lhcbFillColour);
  lhcbStyle->SetCanvasColor(lhcbCanvasColour);
  //lhcbStyle->SetTitleFillColor(lhcbTitleBackColour);

  // large
  //lhcbStyle->SetCanvasDefH(850);
  //lhcbStyle->SetCanvasDefW(1050);
  // medium
  lhcbStyle->SetCanvasDefH(566);
  lhcbStyle->SetCanvasDefW(699);

  // set the paper & margin sizes
  //lhcbStyle->SetPaperSize(20,26);
  //lhcbStyle->SetPadTopMargin(0.05);
  //lhcbStyle->SetPadRightMargin(0.05); // increase for colz plots!!
  //lhcbStyle->SetPadBottomMargin(0.16);
  //lhcbStyle->SetPadLeftMargin(0.14);

  // canvas options
  lhcbStyle->SetCanvasBorderSize(0);
  lhcbStyle->SetCanvasBorderMode(0);

  // use large fonts
  lhcbStyle->SetTextFont(lhcbFont);
  lhcbStyle->SetTextSize(lhcbTextSize);
  lhcbStyle->SetLabelFont(lhcbFont,"x");
  lhcbStyle->SetLabelFont(lhcbFont,"y");
  lhcbStyle->SetLabelFont(lhcbFont,"z");
  lhcbStyle->SetLabelSize(lhcbLabelSize,"x");
  lhcbStyle->SetLabelSize(lhcbLabelSize,"y");
  lhcbStyle->SetLabelSize(lhcbLabelSize,"z");
  lhcbStyle->SetTitleFont(lhcbFont);
  lhcbStyle->SetTitleSize(lhcbAxisLabelSize,"x");
  lhcbStyle->SetTitleSize(lhcbAxisLabelSize,"y");
  lhcbStyle->SetTitleSize(lhcbAxisLabelSize,"z");
  lhcbStyle->SetTitleOffset(lhcbAxisLabelOffset,"x");
  lhcbStyle->SetTitleOffset(lhcbAxisLabelOffset,"y");

  lhcbStyle->SetTitleColor(kWhite);
  lhcbStyle->SetTitleFillColor(kWhite);
  lhcbStyle->SetTitleColor(kBlack);
  lhcbStyle->SetTitleBorderSize(0);
  lhcbStyle->SetTitleTextColor(kBlack);

  // set title position
  lhcbStyle->SetTitleX(0.01);
  lhcbStyle->SetTitleY(0.985);
  lhcbStyle->SetTitleW(0.98);
  // turn off Title box
  lhcbStyle->SetTitleBorderSize(0);
  lhcbStyle->SetTitleTextColor(lhcbForeColour);
  lhcbStyle->SetTitleColor(lhcbForeColour);

  // use bold lines and markers
  lhcbStyle->SetLineWidth(lhcbWidth);
  lhcbStyle->SetFrameLineWidth(lhcbWidth);
  lhcbStyle->SetHistLineWidth(lhcbWidth);
  lhcbStyle->SetFuncWidth(lhcbWidth);
  lhcbStyle->SetGridWidth(lhcbWidth);
  lhcbStyle->SetLineStyleString(2,"[12 12]"); // postscript dashes
  lhcbStyle->SetMarkerStyle(lhcbMarkerType);
  lhcbStyle->SetMarkerSize(lhcbMarkerSize);

  // label offsets
  lhcbStyle->SetLabelOffset(0.015);

  // by default, do not display histogram decorations:
  //lhcbStyle->SetOptStat(0);
  lhcbStyle->SetOptStat(1111);
  //lhcbStyle->SetOptTitle(0);
  //lhcbStyle->SetOptFit(0);
  lhcbStyle->SetOptFit(1011); // show probability, parameters and errors

  // look of the statistics box:
  lhcbStyle->SetStatBorderSize(1);
  lhcbStyle->SetStatFont(lhcbFont);
  lhcbStyle->SetStatFontSize(lhcbStatFontSize);
  lhcbStyle->SetStatX(0.9);
  lhcbStyle->SetStatY(0.9);
  lhcbStyle->SetStatW(lhcbStatBoxWidth);
  lhcbStyle->SetStatH(lhcbStatBoxHeight);

  // Style for 2D zcol plots
  const Int_t NRGBs = 5;
  const Int_t NCont = 255;
  Double_t stops[NRGBs] = { 0.00, 0.34, 0.61, 0.84, 1.00 };
  Double_t red[NRGBs]   = { 0.00, 0.00, 0.87, 1.00, 0.51 };
  Double_t green[NRGBs] = { 0.00, 0.81, 1.00, 0.20, 0.00 };
  Double_t blue[NRGBs]  = { 0.51, 1.00, 0.12, 0.00, 0.00 };
  TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
  lhcbStyle->SetNumberContours(NCont);

  // put tick marks on top and RHS of plots
  lhcbStyle->SetPadTickX(1);
  lhcbStyle->SetPadTickY(1);

  // histogram divisions: only 5 in x to avoid label overlaps
  //lhcbStyle->SetNdivisions(505,"x");
  //lhcbStyle->SetNdivisions(510,"y");

  gROOT->SetStyle("lhcbStyle");
  gROOT->ForceStyle();

}
