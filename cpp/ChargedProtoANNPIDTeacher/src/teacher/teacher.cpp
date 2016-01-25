
// STL
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <cmath>
#include <sstream>
#include <stdlib.h>
#include <time.h>

// ROOT
#include <TSystem.h>
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TProfile.h>
#include <TH2D.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TGraphErrors.h>
#include <TText.h>
#include <TLegend.h>
#include <TCut.h>
#include <TEntryList.h>
#include "Compression.h"

// Local
#include "NodeTypeMapping.h"
#include "ChargedProtoANNPIDTeacher/rootstyle.h"
#include "ChargedProtoANNPIDTeacher/PrintCanvas.h"
#include "NTupleReader.h"

// NeuroBayes
#ifdef ENABLENB
#include "NeuroBayesTeacher.hh"
#include "NeuroBayesExpert.hh"
#else
class NeuroBayesTeacher { };
class Expert            { };
#endif

// TMVA
#include "TMVA/Config.h"
#include "TMVA/Factory.h"
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"

// boost
#include "boost/assign/list_of.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/numeric/conversion/bounds.hpp"
#include "boost/limits.hpp"
#include "boost/format.hpp"
#include "boost/algorithm/string.hpp"
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

//static const std::pair<int,int> canvasDims(1024,768);
static const std::pair<int,int> canvasDims(1280,1024);

static const double GeV = 1000;

// Info string
std::string extraHistoInfo;

// Ntuple reader
NTupleReader * tuple_reader = NULL;

// background types
std::string backgroundTypes;
// ID type
std::string particleType;
// track type
std::string trackType;
// Comb DLL name
std::string combDllVar;

// get time string
const std::string getTime()
{
  time_t     now = time(0);
  struct tm  tstruct;
  char       buf[80];
  tstruct = *localtime(&now);
  // Visit http://www.cplusplus.com/reference/clibrary/ctime/strftime/
  // for more information about date/time format
  strftime( buf, sizeof(buf), "%Y-%m-%d %X ", &tstruct );
  return buf;
}

// MVA parameters
std::map<std::string,std::string> params;

class MaxMin
{
public:
  MaxMin( const double& _max = -999,
          const double& _min =  999 ) : min(_min), max(_max) { }
public:
  double min,max;
};

typedef std::map<std::string,MaxMin> InputMaxMin;

const MaxMin& absLimits( const std::string & var )
{
  static const InputMaxMin limits = boost::assign::map_list_of
    ( "TrackP",    MaxMin(100*GeV,0) )
    ( "TrackPt",   MaxMin(10*GeV,0)  )
    ( "CombDLLe",  MaxMin(20,-20)    )
    ( "CombDLLmu", MaxMin(20,-20)    )
    ( "CombDLLpi", MaxMin(150,-150)  )
    ( "CombDLLk",  MaxMin(150,-150)  )
    ( "CombDLLp",  MaxMin(150,-150)  )
    ;
  const InputMaxMin::const_iterator iS = limits.find(var);
  if ( iS != limits.end() )
  {
    return iS->second;
  }
  else
  {
    static const MaxMin extremes( boost::numeric::bounds<double>::highest(),
                                  boost::numeric::bounds<double>::lowest() );
    return extremes;
  }
}

// Variables that should always be read in (for selection etc.)
const std::vector<std::string> alwaysVars =
  boost::assign::list_of
  ("NumRich2Hits")
  ("TrackType")
  ("TrackP")("TrackPt")
  ("TrackChi2PerDof")
  ("TrackLikelihood")
  ("TrackGhostProbability")
  ("MuonIsLooseMuon")("MuonIsMuon")
  ("CombDLLe")("CombDLLmu")("CombDLLk")("CombDLLp")
  ("RichUsedAero")("RichUsedR1Gas")("RichUsedR2Gas")
  ("InAccEcal")
  ;

const unsigned int allPIDtype(999999);

typedef std::map<std::string,unsigned int> StringToInt;
const StringToInt& particlePDGcodes()
{
  static const StringToInt s_particlePDGcodes = boost::assign::map_list_of
    ( "All", allPIDtype )
    ( "Ghost",    0  )
    ( "Electron", 11 )
    ( "Muon",     13 )
    ( "Pion",    211 )
    ( "Kaon",    321 )
    ( "Proton", 2212 )
    ;
  return s_particlePDGcodes;
}

typedef std::map<unsigned int,std::string> IntToString;
const IntToString& invParticlePDGcodes()
{
  static const IntToString s_invParticlePDGcodes =
    boost::assign::map_list_of
    ( allPIDtype, "All"    )
    ( 0,          "Ghost"  )
    ( 11,       "Electron" )
    ( 13,       "Muon"     )
    ( 211,      "Pion"     )
    ( 321,      "Kaon"     )
    ( 2212,     "Proton"   )
    ;
  return s_invParticlePDGcodes;
}

static std::map<std::string,std::string> combDlls =
  boost::assign::map_list_of
  ( "Electron", "CombDLLe"  )
  ( "Muon",     "CombDLLmu" )
  ( "Pion",     "CombDLLpi" )
  ( "Kaon",     "CombDLLk"  )
  ( "Proton",   "CombDLLp"  )
  ( "Ghost",    "TrackGhostProbability" )
  ;

// Mapping between PID type and PDG code
int particlePDGcodes( const std::string & id )
{
  StringToInt::const_iterator it = particlePDGcodes().find(id);
  if ( it == particlePDGcodes().end() )
  {
    std::cerr << getTime() << "Unknown Particle type " << id << std::endl;
    return -1;
  }
  return it->second;
}

// Mapping between PID type and PDG code
const std::string& invParticlePDGcodes( const unsigned int id )
{
  IntToString::const_iterator it = invParticlePDGcodes().find(id);
  if ( it == invParticlePDGcodes().end() )
  {
    std::cerr << getTime() << "Unknown Particle type " << id << std::endl;
    static const std::string s_nullS = "";
    return s_nullS;
  }
  return it->second;
}

inline double checkValue( const double val )
{
  return ( !std::isnan(val) && !std::isinf(val) ? val : 0.0 );
}

// Calculate the poison error
inline double poisError( const double top, const double bot )
{
  const double val = ( bot > 1e-5 && top > 1e-5 && bot > top ?
                       std::sqrt( std::fabs( (top/bot) * (1.0-top/bot)/bot ) ) : 0.0 );
  return checkValue( val );
}

// Get eff value
inline double getEff( const double top, const double bot )
{
  const double val = ( top>1e-5 && bot>1e-5 ? top/bot : 0.0 );
  return checkValue( val );
}

// Track selection
bool trackTypeSel( const Float_t tType )
{
  bool OK = true;
  if      ( trackType == "Long"       && fabs(tType-3)>0.1 ) OK = false;
  else if ( trackType == "Upstream"   && fabs(tType-4)>0.1 ) OK = false;
  else if ( trackType == "Downstream" && fabs(tType-5)>0.1 ) OK = false;
  return OK;
}

bool createTCut( const std::string & cuts, TCut & sel )
{
  bool ok = true;

  // Open the cuts text file
  std::ifstream cutStream( cuts.c_str() );

  // If OK process
  if ( !cutStream.is_open() )
  {
    ok = false;
    std::cerr << getTime() << "Failed to open " << cuts << std::endl;
  }
  else
  {

    std::cout << getTime() << "Creating selection from '" << cuts << "'" << std::endl;

    std::string fullSelection,cut;
    while ( std::getline(cutStream,cut) )
    {
      if ( cut.empty() ) continue;
      if ( cut.find("#") == std::string::npos )
      {
        std::cout << getTime() << "  -> " << cut << std::endl;
        if ( !fullSelection.empty() ) { fullSelection += " && "; }
        fullSelection += "(" + cut + ")";
      }
      else
      { // Comment, so just print
        std::cout << getTime() << " -> " << cut << std::endl;
      }
    }

    sel = TCut( fullSelection.c_str() );

  }

  cutStream.close();

  return ok;
}

bool mcTrackSel( const std::string& sel,
                 TCut & cut )
{
  bool OK = true;
  if      ( "BTracksOnly" == sel )
  {
    std::cout << getTime() << "Will use only tracks from B decays" << std::endl;
    cut = TCut( "MCFromB" );
  }
  else if ( "DTracksOnly" == sel )
  {
    std::cout << getTime() << "Will use only tracks from D decays" << std::endl;
    cut = TCut( "MCFromD" );
  }
  else if ( "AllTracksInEvent" == sel )
  {
    std::cout << getTime() << "Will use all tracks, regardless of parent type"
              << std::endl;
    cut = TCut();
  }
  else
  {
    std::cerr << getTime() << "ERROR : Unknown MC track selection " << sel << std::endl;
    OK = false;
  }
  return OK;
}

inline double dllMin( const std::string & id )
{
  return ( id == "Electron" ?  -15.0 :
           id == "Muon"     ?  -15.0 :
           id == "Pion"     ? -100.0 :
           id == "Kaon"     ?  -50.0 :
           id == "Proton"   ?  -50.0 :
           id == "Ghost"    ?    0.0 :
           -25.0 );
}

inline double dllMax( const std::string & id )
{
  return ( id == "Electron" ?  15.0 :
           id == "Muon"     ?  15.0 :
           id == "Pion"     ? 100.0 :
           id == "Kaon"     ?  50.0 :
           id == "Proton"   ?  50.0 :
           id == "Ghost"    ?   1.0 :
           25.0 );
}

inline double combDLL()
{
  return ( particleType != "Pion" ?
           tuple_reader->variable(combDllVar) :
           ( -tuple_reader->variable("CombDLLe")
             -tuple_reader->variable("CombDLLmu")
             -tuple_reader->variable("CombDLLk")
             -tuple_reader->variable("CombDLLp") )
           );
}

class PlotLabel
{
public:
  PlotLabel( const std::string& _text, double _x, double _y )
    : text(_text), x(_x), y(_y) { }
  PlotLabel( const double _cut, double _x, double _y )
    : x(_x), y(_y)
  {
    std::ostringstream t;
    t << boost::format("%6.2f") % _cut;
    text = t.str();
  }
public:
  std::string text;
  double x,y;
};
typedef std::vector<PlotLabel> PlotLabels;

class CutProfile
{
public:
  CutProfile( TProfile* _profile = NULL, double _cut = 0 )
    : profile(_profile), cut(_cut) { }
public:
  TProfile* profile;
  double cut;
};

void addCutValues( const PlotLabels& mva_plotLabels,
                   const PlotLabels& dll_plotLabels )
{
  TText text;
  text.SetTextSize(0.025);
  text.SetTextColor(kBlue);
  for ( PlotLabels::const_iterator iMVALabel = mva_plotLabels.begin();
        iMVALabel != mva_plotLabels.end(); ++iMVALabel )
  {
    text.DrawText( iMVALabel->x, iMVALabel->y, iMVALabel->text.c_str() );
  }
  text.SetTextColor(kRed);
  for ( PlotLabels::const_iterator iDLLLabel = dll_plotLabels.begin();
        iDLLLabel != dll_plotLabels.end(); ++iDLLLabel )
  {
    text.DrawText( iDLLLabel->x, iDLLLabel->y, iDLLLabel->text.c_str() );
  }
}

void setGraphLimits( TGraphErrors * g,
                     const double xMin,
                     const double xMax,
                     const double yMin,
                     const double yMax )
{
  if ( g )
  {
    if ( g->GetXaxis() )
    {
      g->GetXaxis()->SetLimits( xMin, xMax );
    }
    if ( g->GetHistogram() )
    {
      g->GetHistogram()->SetMinimum( yMin );
      g->GetHistogram()->SetMaximum( yMax );
    }
  }
}

inline void removeSpaces( std::string & str )
{
  boost::erase_all(str," ");
}

void getMinMaxEffPur( const std::string & trackType,
                      const std::string & particleType,
                      double & minPur,
                      double & maxPur,
                      double & minEffForPur,
                      double & maxEffForPur )
{
  minPur       = 0.1;
  maxPur       = 100.;
  minEffForPur = 0.1;
  maxEffForPur = 110.;
  if ( "Long" == trackType )
  {
    if ( "Pion" == particleType )
    {
      minPur = 60.0;
    }
  }
  else if ( "Downstream" == trackType )
  {
    if ( "Pion" == particleType )
    {
      minPur = 10.0;
    }
    else if ( "Ghost" == particleType )
    {
      minPur = 60.0;
    }
  }
  else if ( "Upstream" == trackType )
  {
    if ( "Pion" == particleType )
    {
      minPur = 40.0;
    }
    else if ( "Ghost" == particleType )
    {
      minPur = 20.0;
    }
  }
}

typedef std::map<std::string,CutProfile> StringTProfileMap;

void makeEffVVarPlots( const std::string& mvaDir,
                       StringTProfileMap & mEff,
                       StringTProfileMap & mMis )
{
  TCanvas * c = new TCanvas( ("EffVMomentum-"+mvaDir).c_str(),
                             (trackType+" "+particleType+" EffVMomentum | "
                              + extraHistoInfo ).c_str(),
                             canvasDims.first, canvasDims.second );
  c->SetGrid();
  for ( StringTProfileMap::const_iterator iC = mEff.begin();
        iC != mEff.end(); ++iC )
  {
    const std::string & cut = (*iC).first;
    TProfile * eff = (*iC).second.profile;
    TProfile * mis = mMis[cut].profile;
    if ( eff && mis )
    {
      const bool misFirst = ( mis->GetBinContent(mis->GetMaximumBin()) >
                              eff->GetBinContent(eff->GetMaximumBin()) );
      eff->SetStats(0);
      mis->SetStats(0);
      mis->SetMarkerColor(kRed);
      mis->SetLineColor(kRed);
      eff->SetMarkerColor(kBlue);
      eff->SetLineColor(kBlue);
      eff->SetMaximum(100);
      eff->SetMinimum(0);
      mis->SetMaximum(100);
      mis->SetMinimum(0);
      mis->SetMarkerStyle(kFullTriangleUp);
      eff->SetMarkerStyle(kFullCircle);
      (  misFirst ? mis : eff ) -> Draw("E");
      ( !misFirst ? mis : eff ) -> Draw("E SAME");
      TLegend * legend = new TLegend(0.8,0.9,0.9,0.85);
      legend->SetTextSize(0.02);
      legend->SetMargin(0.1);
      legend->AddEntry( eff, ("True " + particleType).c_str(), "p" );
      legend->AddEntry( mis, ("Fake " + particleType).c_str(), "p" );
      legend->Draw();
      printCanvas( c, mvaDir+"/"+mvaDir+"_"+trackType+"_"+particleType+"-"+cut );
    }
  }
}

std::string getVarAxisLabel( const std::string & var )
{
  typedef std::map<std::string,std::string> Map;
  static Map labels = boost::assign::map_list_of
    ( "TrackP" , "Momentum / MeV/c" )
    ( "TrackPt", "Transverse Momentum / MeV/c" )
    ;
  Map::const_iterator i = labels.find(var);
  return ( i == labels.end() ? var : i->second );
}

template < class TYPE >
inline TYPE getParam( const std::string & name, const TYPE defValue )
{
  std::map<std::string,std::string>::const_iterator i = params.find(name);
  return ( i != params.end() ?
           boost::lexical_cast<TYPE>( i->second ) :
           defValue );
}

bool applyRich2HitsWeight()
{
  static TRandom rand;
  const double x = tuple_reader->variable("NumRich2Hits");
  const double w = (  0.00108528 +
                      ( -1.0822e-05  * x           ) +
                      ( 3.51537e-08  * x*x         ) +
                      ( -4.6664e-11  * x*x*x       ) +
                      ( 3.16421e-14  * x*x*x*x     ) +
                      ( -8.69538e-18 * x*x*x*x*x   ) +
                      ( 1.14595e-21  * x*x*x*x*x*x ) );
  const double r = rand.Uniform(1.0);
  return ( r < w );
}

int main(int argc, char** argv)
{
  if ( argc != 7 )
  {
    std::cout << getTime() << "Wrong number of arguments" << std::endl;
    return 1;
  }

  // ---------------------------------------------------------------------------------
  // Initialise
  // ---------------------------------------------------------------------------------

  // needed to fix some odd dictionary problem... even though ROOT
  // claims it is already loaded ... :(
  gSystem->Load("libTree");
  // root style
  setStyle();
  // turn off info messages
  gROOT->ProcessLine("gErrorIgnoreLevel = kWarning;");

  // Load environment vars
  char * mvaConfigNameChar = getenv("NNCONFIGNAME");
  if ( !mvaConfigNameChar )
  {
    std::cerr << getTime() << "Failed to read environment variable NNCONFIGNAME" << std::endl;
    return 1;
  }
  const std::string mvaConfigName(mvaConfigNameChar);

  // Running options
  const std::string doTraining(argv[5]), doEvaluation(argv[6]);

  // open network config file
  const std::string configFile(argv[1]);
  std::cout << getTime() << "Network Config file      = " << configFile << std::endl;
  std::ifstream config(configFile.c_str());
  if ( !config.is_open() )
  {
    std::cerr << getTime() << "Error opening network configuration file" << std::endl;
    return 1;
  }

  // open training config file
  const std::string trainingParams(argv[2]);
  std::cout << getTime() << "Training parameters file = " << trainingParams << std::endl;
  std::ifstream training(trainingParams.c_str());
  if ( !training.is_open() )
  {
    std::cerr << getTime() << "Error opening training parameters file" << std::endl;
    return 1;
  }

  // open input data config file
  const std::string dataParams(argv[3]);
  std::cout << getTime() << "Input Data file = " << dataParams << std::endl;
  std::ifstream dataFiles(dataParams.c_str());
  if ( !dataFiles.is_open() )
  {
    std::cerr << getTime() << "Error opening Input Data file" << std::endl;
    return 1;
  }

  // open NB specific config file
  const std::string nbParamsFile(argv[4]);
  std::cout << getTime() << "Neurobayes parameters file = " << nbParamsFile << std::endl;
  std::ifstream nbParms(nbParamsFile.c_str());
  if ( !nbParms.is_open() )
  {
    std::cout << getTime() << "Error opening NB parameters file" << std::endl;
    return 1;
  }

  // Read the particle type
  config >> particleType;
  particleType[0] = toupper(particleType[0]);

  const int iPIDtype = particlePDGcodes(particleType);
  if ( -1 == iPIDtype ) { return 1; }
  combDllVar = combDlls[particleType];

  // Read the track type
  config >> trackType;

  // Track Selection
  std::string trackSelFile;
  config >> trackSelFile;

  // Background type
  training >> backgroundTypes;
  backgroundTypes[0] = toupper(backgroundTypes[0]);
  const unsigned int iBackPID = particlePDGcodes(backgroundTypes);

  // Ghost Treatment for training
  std::string ghostTreatmentTraining;
  training >> ghostTreatmentTraining;
  const bool keepGhostsTraining = "NoGhosts" != ghostTreatmentTraining;

  // Ghost Treatment for evaluation
  std::string ghostTreatmentEval;
  training >> ghostTreatmentEval;
  const bool keepGhostsEval = "NoGhosts" != ghostTreatmentEval;

  // MC Track training selection
  std::string mcTrackSelTraining;
  training >> mcTrackSelTraining;

  // MC Track eval selection
  std::string mcTrackSelEval;
  training >> mcTrackSelEval;

  const std::string ghostTreatSummary =
    ( "Train:" + ghostTreatmentTraining + "-Eval:" + ghostTreatmentEval );

  // Signal/Background mix
  std::string trainingMix;
  training >> trainingMix;

  // Reweighting
  std::string reweightOpt;
  training >> reweightOpt;

  // Read the network type
  std::string mvaType;
  config >> mvaType;

  if ( "NeuroBayes" != mvaType && "TMVA" != mvaType )
  {
    std::cerr << getTime() << "Unknown Network Type " << mvaType << std::endl;
    return 1;
  }

#ifndef ENABLENB
  if ( "NeuroBayes" == mvaType )
  {
    std::cerr << getTime() << "NeuroBayes not supported" << std::endl;
    return 1;
  }
#endif

  std::cout << getTime() << "Particle type    = " << particleType << std::endl;
  std::cout << getTime() << "Background types = " << backgroundTypes << std::endl;
  std::cout << getTime() << "Training Mix     = " << trainingMix << std::endl;
  std::cout << getTime() << "Track type       = " << trackType << std::endl;
  std::cout << getTime() << "Track PresSel    = " << trackSelFile << std::endl;
  std::cout << getTime() << "Network type     = " << mvaType << std::endl;
  std::cout << getTime() << "Ghost treatment  = " << ghostTreatmentTraining
            << " " << ghostTreatmentEval << std::endl;
  std::cout << getTime() << "MC Track sel.    = " << mcTrackSelTraining
            << " " << mcTrackSelEval << std::endl;
  std::cout << getTime() << "Reweighting      = " << reweightOpt << std::endl;

  // sanity checks
  if ( particleType == backgroundTypes )
  {
    std::cerr << getTime() << "Background and PID types the same " << particleType << " !!!"
              << std::endl;
    return 1;
  }
  if ( "Ghost" == particleType &&
       ( "NoGhosts" == ghostTreatmentTraining ||
         "NoGhosts" == ghostTreatmentEval     || 
         "BTracksOnly" == mcTrackSelTraining ) )
  {
    std::cerr << getTime() << "Cannot train ghost ID network whilst rejecting ghosts !!!"
              << std::endl;
    return 1;
  }

  // Read network parameters file name
  std::string paramFileName;
  config >> paramFileName;

  std::cout << getTime() << "Output file      = " << paramFileName << std::endl;

  // Read the list of inputs
  std::string input;
  std::vector<std::string> inputs;
  while ( config >> input )
  {
    if ( !input.empty() )
    {
      if ( input.find("#") == std::string::npos )
      {
        inputs.push_back(input);
      }
      else
      {
        std::cout << getTime() << "Skipping input " << input << std::endl;
      }
    }
  }

  // Build the track preselection
  TCut trackSel;
  if ( !createTCut(trackSelFile,trackSel) )
  {
    std::cerr << getTime() << "Failed to construct track selection" << std::endl;
    return 1;
  }

  // MC Track training sel
  TCut mcTrackTrainSel;
  if ( !mcTrackSel( mcTrackSelTraining, mcTrackTrainSel ) ) { return 1; }

  // read in the list of data files
  std::vector<std::string> trainingFiles;
  {
    std::string fileN;
    while ( std::getline(dataFiles,fileN) )
    {
      if ( fileN.empty() || fileN.find("#") != std::string::npos ) continue;
     trainingFiles.push_back(fileN);
    }
    std::cout << getTime() << "Using " << trainingFiles.size() << " data files" << std::endl;
    if ( trainingFiles.size() < 2 )
    {
      std::cerr << getTime() << "ERROR : Missing datafiles" << std::endl;
      return 1;
    }
  }

  // Read the training parameters

  unsigned int nTrainingTracks;
  training >> nTrainingTracks;
  std::cout << getTime() << "Training sample size   = " << nTrainingTracks << std::endl;

  unsigned int nEvalTracks;
  training >> nEvalTracks;
  std::cout << getTime() << "Evaluation sample size = " << nEvalTracks << std::endl;

  // read the MVA specific training parameters
  std::cout << getTime() << "Reading MVA parameters" << std::endl;
  std::string param;
  while ( std::getline(nbParms,param) )
  {
    if ( !param.empty() && param.find("#") == std::string::npos )
    {
      boost::regex expr("(.*)=(.*)");
      boost::cmatch matches;
      if ( boost::regex_match( param.c_str(), matches, expr ) )
      {
        std::string type = matches[1];
        boost::erase_all(type," ");
        std::string val  = matches[2];
        boost::erase_all(val," ");
        std::cout << getTime() << " -> " << type << " = " << val << std::endl;
        params[type] = val;
      }
      else
      {
        std::cerr << getTime() << "ERROR : Failed to parse " << param << std::endl;
        return 1;
      }
    }
  }

  // close the file streams
  config.close();
  training.close();
  dataFiles.close();
  nbParms.close();

  // The location of the tuple in the ROOT file
  const std::string tupleLocation = "ANNPID.Tuple/annInputs";

  // ----------------------------------------------------------------------------------
  // Configure the teacher
  // ----------------------------------------------------------------------------------

  // Number of inputs and hidden nodes
  const double layerTwoScale = getParam<double>("HiddenLayerScaleFactor",1.2);
  std::cout << getTime() << "Node layer two scale = " << layerTwoScale << std::endl;
  const int nvar         = inputs.size(); // number of input variables
  const int nHiddenNodes = (int)(layerTwoScale*nvar);
  std::cout << getTime() << "Input layer has " << nvar+1 << " nodes, hidden layer has "
            << nHiddenNodes << " nodes" << std::endl;

  // string for histograms with all the interesting information
  std::string tmpTrackSel = trackSelFile;
  boost::erase_all(tmpTrackSel,".txt");
  extraHistoInfo = ( ghostTreatSummary + " | Bck. " + backgroundTypes + " " +
                     trainingMix + " " + mcTrackSelTraining + " " + reweightOpt +
                     " | " + mvaConfigName );

  // NB parameters
  const std::string nbTask      = getParam<std::string>("NBTASK","CLA");
  const std::string nbShape     = getParam<std::string>("NBSHAPE","DIAG");
  const std::string nbReg       = getParam<std::string>("NBREG","ALL");
  const std::string nbLoss      = getParam<std::string>("NBLOSS","ENTROPY");
  const std::string nbMethod    = getParam<std::string>("NBMETHOD","BFGS");
  const int         nbLearnDiag = getParam<int>("NBLEARNDIAG",1);
  const int         nbPre       = getParam<int>("NBPRE",612);

  // TMVA parameters
  const std::string tmvaMethod           = getParam<std::string>("TMVAMETHOD","MLP");
  const std::string tmvaVarTransform     = getParam<std::string>("TMVAVARTRANSFORM","None");
  const std::string tmvaMLPNeuronType    = getParam<std::string>("TMVAMLPNEURONTYPE","sigmoid");
  const std::string tmvaMLPMethod        = getParam<std::string>("TMVAMLPMETHOD","BP");
  const std::string tmvaMLPNCycles       = getParam<std::string>("TMVAMLPNCYCLES","500");
  const std::string tmvaMLPEstimatorType = getParam<std::string>("TMVAMLPESTIMATORTYPE","CE");
  const std::string tmvaBDTBoostType     = getParam<std::string>("TMVABDTBOOSTTYPE","AdaBoost");
  const std::string tmvaBDTNTrees        = getParam<std::string>("TMVABDTNTREES","800");
  const std::string tmvaBDTPruneMethod   = getParam<std::string>("TMVABDTPRUNEMETHOD","NoPruning");
  const std::string tmvaBDTMaxTreeDepth  = getParam<std::string>("TMVABDTMAXTREEDEPTH","3");
  const float       tmvaValidationFrac   = getParam<float>("TMVAVALIDATIONFRAC",0.3);

  // NULL pointers until we know what to do ...
#ifdef ENABLENB
  NeuroBayesTeacher * nbTeacher = NULL;
#endif
  TFile* tmvaHistoFile          = NULL;
  TMVA::Factory * tmvaFactory   = NULL;
  std::string mvaTextForFile    = mvaConfigName + "-" + reweightOpt;

  if ( "NeuroBayes" == mvaType )
  {
    std::ostringstream nbParamSummary;
    nbParamSummary << nbShape << " " << nbLoss << " LD" << nbLearnDiag
                   << " " << nbMethod << " " << nbPre << " SF" << layerTwoScale;
    extraHistoInfo += " | " + nbParamSummary.str();
    mvaTextForFile += "-" + nbParamSummary.str();
  }
  else if ( "TMVA" == mvaType )
  {
    // Load the library
    TMVA::Tools::Instance();
    std::ostringstream tmvaParamSummary;
    tmvaParamSummary << tmvaMethod << " " << tmvaVarTransform;
    if ( "MLP" == tmvaMethod )
    {
      tmvaParamSummary << " " << tmvaMLPMethod
                       << " NCycles" << tmvaMLPNCycles
                       << " " << tmvaMLPEstimatorType
                       << " " << tmvaMLPNeuronType
                       << " SF" << layerTwoScale;
    }
    else if ( "BDT" == tmvaMethod )
    {
      tmvaParamSummary << " " << tmvaBDTBoostType
                       << " NTrees" << tmvaBDTNTrees
                       << " MaxDepth" << tmvaBDTMaxTreeDepth
                       << " " << tmvaBDTPruneMethod;
    }
    extraHistoInfo += " | " + tmvaParamSummary.str();
    mvaTextForFile += "-" + tmvaParamSummary.str();
  }
  boost::replace_all(mvaTextForFile," ","-");

  if ( doTraining == "Yes" )
  {

    // Set up the trainer
#ifdef ENABLENB
    if ( "NeuroBayes" == mvaType )
    {

      const float nbMom = getParam<float>("NBMOM",0.0f);

      // create NeuroBayes instance
      nbTeacher = NeuroBayesTeacher::Instance();

      nbTeacher->NB_DEF_NODE1(nvar+1);       // nodes in input layer, leave this alone! (+1 is correct...)
      nbTeacher->NB_DEF_NODE2(nHiddenNodes); // nodes in hidden layer, you may play with this one
      nbTeacher->NB_DEF_NODE3(1);            // nodes in output layer

      nbTeacher->NB_DEF_TASK(nbTask.c_str());
      nbTeacher->NB_DEF_SHAPE(nbShape.c_str());
      nbTeacher->NB_DEF_REG(nbReg.c_str());
      nbTeacher->NB_DEF_LOSS(nbLoss.c_str());
      nbTeacher->NB_DEF_METHOD(nbMethod.c_str());
      nbTeacher->NB_DEF_LEARNDIAG(nbLearnDiag);
      nbTeacher->NB_DEF_PRE(nbPre);
      nbTeacher->NB_DEF_MOM(nbMom);

      // output expertise file
      nbTeacher->SetOutputFile(paramFileName.c_str());

      // Set the variable prepro-flags
      int i(0);
      for ( std::vector<std::string>::const_iterator iIn = inputs.begin();
            iIn != inputs.end(); ++iIn, ++i )
      {
        NeuroBayesNodeTypeMap::const_iterator iN = s_NBnodeTypeMap.find(*iIn);
        if ( iN == s_NBnodeTypeMap.end() )
        {
          std::cerr << getTime() << "Unknown input " << *iIn << std::endl;
          return 1;
        }
        std::cout << getTime() << "Input " << 2+i << " " << *iIn
                  << " PreproFlag " << iN->second << std::endl;
        nbTeacher->SetIndividualPreproFlag(i,iN->second);
      }

    }
#endif
    if ( "TMVA" == mvaType )
    {

      // Output ROOT file
      const std::string tmvaHistoFileName = trackType+"-"+particleType+"-TMVA.root";
      std::cout << getTime() << "TMVA Output ROOT file " << tmvaHistoFileName << std::endl;
      tmvaHistoFile = TFile::Open( tmvaHistoFileName.c_str(), "RECREATE" );
      if ( !tmvaHistoFile )
      {
        std::cerr << getTime() << "Failed to open TMVA output ROOT file " << tmvaHistoFileName << std::endl;
        return 1;
      }
      tmvaHistoFile->SetCompressionSettings(ROOT::CompressionSettings(ROOT::kLZMA,5));
      tmvaHistoFile->cd((tmvaHistoFileName+":/").c_str());

      if ( tmvaValidationFrac > 0 && "BDT" == tmvaMethod && "NoPruning" != tmvaBDTPruneMethod )
      {
        nTrainingTracks = (unsigned int)( (double)nTrainingTracks / (double)(1.0-tmvaValidationFrac) );
        std::cout << getTime() << "Scaling # Training Tracks by " << 1.0+tmvaValidationFrac
                  << " to " << nTrainingTracks << std::endl;
      }

      // Factory
      tmvaFactory = new TMVA::Factory( "GlobalPID",
                                       tmvaHistoFile,
                                       "V:!Silent:!Color:!DrawProgressBar:Transformations=I;D;P;G,D" );

      // Set weights dir
      (TMVA::gConfig().GetIONames()).fWeightFileDir = ".";

      // Add variables
      int i(0);
      for ( std::vector<std::string>::const_iterator iIn = inputs.begin();
            iIn != inputs.end(); ++iIn, ++i )
      {
        std::cout << getTime() << "Input " << *iIn
                  << " Type " << getTMVANodeType(*iIn) << std::endl;
        tmvaFactory->AddVariable( (*iIn).c_str(), getTMVANodeType(*iIn) );
      }

    }

  }

  // -----------------------------------------------------------------------------------------
  // Read in training data
  // -----------------------------------------------------------------------------------------

  // Array/vector for variables
  float InputArray[nvar];
  std::vector<double> InputVector(nvar);

  // Keep tabs on max and min value for each input
  InputMaxMin inputMaxMin;

  // Count tracks
  unsigned int allTracks(0),selectedTracks(0),signalTracks(0),backgroundTracks(0);
  std::map<unsigned int,unsigned int> selectedTracksByType;

  // Loop over training file list as far as required
  std::vector<std::string>::const_iterator iFileName = trainingFiles.begin();
  unsigned int nFilesUsedInTraining(0);
  for ( ; iFileName != trainingFiles.end(); ++iFileName, ++nFilesUsedInTraining )
  {

    // Load the training ntuple
    std::cout << getTime() << "Using training file " << *iFileName << std::endl;
    TFile * training_file = TFile::Open((*iFileName).c_str());
    if ( !training_file )
    {
      std::cerr << getTime() << "Failed to open input ROOT file " << *iFileName << std::endl;
      return 1;
    }
    TTree * training_tree = (TTree*)gDirectory->Get(tupleLocation.c_str());
    if ( !training_tree )
    {
      std::cerr << getTime() << "Failed to open MVA TTree" << std::endl;
      return 1;
    }

    // If TMVA go back to the TMVA file to prevent memory resident warnings
    if ( tmvaHistoFile ) tmvaHistoFile->cd();

    // Apply TCut selection
    std::ostringstream elistname;
    elistname << "training_elist";
    training_tree->Draw( ">> training_elist", trackSel && mcTrackTrainSel, "entrylist" );
    TEntryList * training_tree_elist = (TEntryList*) gDirectory->Get("training_elist");
    if ( !training_tree_elist )
    { std::cerr << getTime() << "ERROR : Problem getting selected entries" << std::endl; return 1; }
    std::cout << getTime() << "Selected " << training_tree_elist->GetN() << " entries from "
              << training_tree->GetEntries()
              << " ( " << 100.0 * training_tree_elist->GetN() / training_tree->GetEntries() << " % )"
              << std::endl;

    // Create a Ntuple reader
    tuple_reader = new NTupleReader( training_tree, inputs);
    // variables that always should be available, even if not part of network inputs
    for ( std::vector<std::string>::const_iterator iV = alwaysVars.begin();
          iV != alwaysVars.end(); ++iV )
    {
      tuple_reader->addVariable( training_tree, *iV );
    }

    // MC type
    Int_t MCParticleType(0);
    TBranch * b_MCParticleType(NULL);
    training_tree->SetBranchAddress( "MCParticleType", &MCParticleType, &b_MCParticleType );

    // Loop over entry list
    training_tree->SetEntryList(training_tree_elist);
    for ( Long64_t iEntry = 0; iEntry < training_tree_elist->GetN(); ++iEntry )
    {
      const int entryN = training_tree->GetEntryNumber(iEntry);
      if ( entryN < 0 ) break;
      training_tree->GetEntry(entryN);
      // if ( (iEntry+1) % (training_tree_elist->GetN()/10) == 0 )
      // {
      //   std::cout << getTime() << "Read entry " << (1+iEntry)
      //             << " (" << 100.0*((float)(1+iEntry)/(float)training_tree_elist->GetN()) << "%)"
      //             << std::endl;
      // }

      // count all tracks considered
      ++allTracks;

      // Track selection
      if ( !trackTypeSel(tuple_reader->variable("TrackType")) ) continue;

      // ghost treatment
      if ( 0 == MCParticleType && !keepGhostsTraining ) continue;

      // true or fake target
      const bool target = ( abs(MCParticleType) == iPIDtype );

      // Background type selection for training
      if ( ! ( target                                 ||
               allPIDtype          ==      iBackPID   ||
               abs(MCParticleType) == (int)iBackPID )  ) continue;

      // signal/background mix
      if ( "EqualMix" == trainingMix )
      {
        if ( signalTracks != backgroundTracks )
        {
          if ( ( target  && signalTracks > backgroundTracks ) ||
               ( !target && signalTracks < backgroundTracks )  ) continue;
        }
      }
      ++( target ? signalTracks : backgroundTracks );

      // Sample events based on # Rich2 hits
      if ( "ReweightRICH2" == reweightOpt && !applyRich2HitsWeight() ) continue;

      // Fill an input array for the teacher
      int ivar = 0;
      for ( std::vector<std::string>::const_iterator iIn = inputs.begin();
            iIn != inputs.end(); ++iIn, ++ivar )
      {
        const double var = tuple_reader->variable(*iIn);
        InputArray[ivar]  = var;
        InputVector[ivar] = var;
        // Max and mins for histograms later on
        MaxMin & maxmin = inputMaxMin[*iIn];
        if ( var > maxmin.max                  ) maxmin.max = var;
        if ( var > -998.0 && var < maxmin.min  ) maxmin.min = var;
      }
      const double dll = combDLL();
      MaxMin & maxmin = inputMaxMin[combDllVar];
      if ( dll > maxmin.max                  ) maxmin.max = dll;
      if ( dll > -998.0 && dll < maxmin.min  ) maxmin.min = dll;

      // Set inputs and target output
#ifdef ENABLENB
      if ( nbTeacher)
      {
        nbTeacher->SetTarget( target ? 1.0 : -1.0 );
        nbTeacher->SetNextInput(nvar,InputArray);
      }
#endif
      if ( tmvaFactory )
      {
        if ( target )
        {
          tmvaFactory->AddSignalTrainingEvent( InputVector, 1.0 );
          tmvaFactory->AddSignalTestEvent    ( InputVector, 1.0 );
        }
        else
        {
          tmvaFactory->AddBackgroundTrainingEvent( InputVector, 1.0 );
          tmvaFactory->AddBackgroundTestEvent    ( InputVector, 1.0 );
        }
      }

      // count tracks
      ++selectedTracks;
      ++(selectedTracksByType[abs(MCParticleType)]);

      // Found enough tracks ?
      if ( selectedTracks >= nTrainingTracks ) break;

    } // loop over ntuple entries

    // delete the tuple reader
    delete tuple_reader;
    tuple_reader = NULL;

    // Delete entry list
    training_tree_elist->Delete();

    // close training root file
    training_file->Close();

    // Found enough tracks ?
    if ( selectedTracks >= nTrainingTracks ) break;

    // Used half of the training files ?
    if ( (nFilesUsedInTraining+1) > (trainingFiles.size()/2) ) break;

  } // loop over data files

  // check min max bounds against absolute limits
  for ( InputMaxMin::iterator iL = inputMaxMin.begin();
        iL != inputMaxMin.end(); ++iL )
  {
    const MaxMin& absL = absLimits( iL->first );
    if ( absL.max < iL->second.max ) iL->second.max = absL.max;
    if ( absL.min > iL->second.min ) iL->second.min = absL.min;
  }

  std::cout << getTime() << "Considered " << allTracks << " tracks for input to MVA training" << std::endl;
  if ( allTracks == 0 ) return 1;
  std::cout << getTime() << "  Sel. Eff. = "
            << 100.0*(double)selectedTracks/(double)allTracks << "%" << std::endl;
  std::cout << getTime() << "Selected " << selectedTracks << " tracks for input to MVA training" << std::endl;
  if ( selectedTracks == 0 ) return 1;
  for ( std::map<unsigned int,unsigned int>::const_iterator iSelT = selectedTracksByType.begin();
        iSelT != selectedTracksByType.end(); ++iSelT )
  {
    std::cout << getTime() << "  " << invParticlePDGcodes(iSelT->first) << " percentage = "
              << 100.0*(double)iSelT->second/(double)selectedTracks << "%" << std::endl;
  }

  // For TMVA, must setup the method here
  if ( tmvaFactory )
  {

    // Prepare the data
    TCut cuts("");
    tmvaFactory->PrepareTrainingAndTestTree( cuts,
                                             "nTrain_Signal=0:nTrain_Background=0:SplitMode=Random:NormMode=NumEvents:V" );

    // TMVA Method name
    std::ostringstream name;
    name << particleType << "_" << trackType << "_TMVA";

    // Sort of TMVA classifier
    if ( "MLP" == tmvaMethod )
    {
      // Construct MLP options
      std::ostringstream opts;
      opts << "H:V:EpochMonitoring:HiddenLayers=" << nHiddenNodes;
      //opts << ":CreateMVAPdfs=True";
      if ( "DEFAULT" != tmvaVarTransform )     { opts << ":VarTransform="   << tmvaVarTransform;     }
      if ( "DEFAULT" != tmvaMLPNCycles )       { opts << ":NCycles="        << tmvaMLPNCycles;       }
      if ( "DEFAULT" != tmvaMLPNeuronType )    { opts << ":NeuronType="     << tmvaMLPNeuronType;    }
      if ( "DEFAULT" != tmvaMLPMethod )        { opts << ":TrainingMethod=" << tmvaMLPMethod;        }
      if ( "DEFAULT" != tmvaMLPEstimatorType ) { opts << ":EstimatorType="  << tmvaMLPEstimatorType; }
      std::cout << getTime() << "TMVA MLP '" << opts.str() << "'" << std::endl;
      tmvaFactory->BookMethod( TMVA::Types::kMLP, name.str().c_str(), opts.str().c_str() );
    }
    else if ( "BDT" == tmvaMethod )
    {
      // BDT opts
      std::ostringstream opts;
      opts << "!H:V:NTrees=" << tmvaBDTNTrees;
      //opts << ":CreateMVAPdfs=True";
      if ( "DEFAULT" != tmvaVarTransform )    { opts << ":VarTransform=" << tmvaVarTransform;    }
      if ( "DEFAULT" != tmvaBDTBoostType )    { opts << ":BoostType="    << tmvaBDTBoostType;    }
      if ( "DEFAULT" != tmvaBDTPruneMethod )  { opts << ":PruneMethod="  << tmvaBDTPruneMethod;  }
      if ( "CostComplexity" == tmvaBDTPruneMethod || "ExpectedError" == tmvaBDTPruneMethod )
      { opts << ":PruneStrength=-1"; }
      if ( "DEFAULT" != tmvaBDTMaxTreeDepth ) { opts << ":MaxDepth="     << tmvaBDTMaxTreeDepth; }
      if ( tmvaValidationFrac > 0 && "NoPruning" != tmvaBDTPruneMethod )
      { opts << ":PruningValFraction=" << tmvaValidationFrac; }
      std::cout << getTime() << "TMVA BDT '" << opts.str() << "'" << std::endl;
      tmvaFactory->BookMethod( TMVA::Types::kBDT, name.str().c_str(), opts.str().c_str() );
    }
    else
    {
      std::cerr << getTime() << "ERROR : Unknown TMVA Method " << tmvaMethod << std::endl;
      return 1;
    }

  }

  // -------------------------------------------------------------------------------------------
  // Train the network
  // -------------------------------------------------------------------------------------------

  // Train the network
  if ( doTraining == "Yes" )
  {
    std::cout << getTime() << "Starting Training." << std::endl;
#ifdef ENABLENB
    if ( nbTeacher )
    {
      nbTeacher->TrainNet();
    }
#endif
    if ( tmvaFactory )
    {
      // Train MVAs using the set of training events
      tmvaFactory->TrainAllMethods();
      // ---- Evaluate all MVAs using the set of test events
      tmvaFactory->TestAllMethods();
      // ----- Evaluate and compare performance of all configured MVAs
      tmvaFactory->EvaluateAllMethods();
      // Cleanup
      if ( tmvaHistoFile ) tmvaHistoFile->Close();
      delete tmvaFactory;
    }
    std::cout << getTime() << "Training Complete :)" << std::endl;
  }

  // -------------------------------------------------------------------------------------------
  // Evaluate the trained network
  // -------------------------------------------------------------------------------------------

  if ( doEvaluation == "Yes" )
  {

    // Open Histograms file
    const std::string histoFileName = trackType+"-"+particleType+"-evaluation.root";
    const std::string histoFileRoot = histoFileName+":/";
    TFile histoFile(histoFileName.c_str(),"recreate");
    histoFile.SetCompressionSettings(ROOT::CompressionSettings(ROOT::kLZMA, 5));
    histoFile.cd(histoFileRoot.c_str());

    // Number of cut values to consider
    const unsigned int nCutsMVA(80), nCutsDLL(150);
    // min and max cuts
    const double mvaMin(0.0), mvaMax(1.0);
    // number of bins in histograms
    const unsigned int nBins(100), nBins2D(50);
    // step sizes
    const double mvastep  = ( mvaMax - mvaMin  ) / (double)(nCutsMVA-1) ;
    const double dllstep = ( dllMax(particleType) - dllMin(particleType) ) / (double)(nCutsDLL-1) ;
    // data storage
    typedef std::pair < unsigned int, unsigned int > SelData;
    typedef std::map  < unsigned int, SelData      > PIDData;
    typedef std::map  < double,       PIDData      > PIDStepData;
    // data storage
    PIDStepData pidStepDataMVA, pidStepDataCombDLL;

    // MVA out cuts for classification
    const std::vector<double> mvaOutCuts =
      ( "EqualMix" == trainingMix
        ?
        boost::assign::list_of
        (0.01)(0.05)(0.1)(0.2)(0.3)(0.4)(0.5)(0.6)(0.7)(0.8)(0.9)(0.95)(0.99)
        :
        boost::assign::list_of
        (0.025)(0.05)(0.075)(0.1)(0.125)(0.15)(0.175)(0.2)(0.3)(0.35)(0.4)(0.45)(0.5)(0.6)(0.7)(0.8)(0.9)(0.95)
        );

    // DLL cuts for classification
    const std::vector<double> dllCuts = boost::assign::list_of
      (-8.0)(-7.0)(-6.0)(-5.5)(-5.0)(-4.5)(-4.0)(-3.5)(-3.0)(-2.5)(-2.0)(-1.5)(-1.0)(-0.5)(0.0)
      (0.5)(1.0)(1.5)(2.0)(2.5)(3.0)(3.5)(4.0)(4.5)(5.0)(5.5)(6.0)(7.0)(8.0);

    // Book some histograms for the input variables
    std::map<std::string,TH1F*> inVarHTrue,inVarHFake;
    std::map<std::string,StringTProfileMap> mvaeffVinVar, mvamisidVinVar, dlleffVinVar, dllmisidVinVar;
    std::map<std::string,bool> mvadirMade, dlldirMade;
    for ( std::vector<std::string>::const_iterator iIn = inputs.begin();
          iIn != inputs.end(); ++iIn )
    {
      std::cout << getTime() << "Making plots for input " << *iIn << std::endl;
      histoFile.cd(histoFileRoot.c_str());
      unsigned int varBins = nBins;
      double maxvar = 1.01 * inputMaxMin[*iIn].max;
      double minvar = 0.99 * inputMaxMin[*iIn].min;
      NeuroBayesNodeTypeMap::const_iterator iN = s_NBnodeTypeMap.find(*iIn);
      // if descrete variable, different histo settings
      if ( iN != s_NBnodeTypeMap.end() &&
           ( iN->second == UnorderedClass ||
             iN->second == OrderedClass ) )
      {
        maxvar  = inputMaxMin[*iIn].max + 0.5;
        minvar  = inputMaxMin[*iIn].min - 0.5;
        varBins = int( inputMaxMin[*iIn].max - inputMaxMin[*iIn].min ) + 1;
      }
      // MVA inputs
      inVarHTrue[*iIn] = new TH1F( ("hTrue_" + (*iIn)).c_str(),
                                   ( (*iIn) + " True " + particleType +
                                     " " + trackType + " | " + extraHistoInfo ).c_str(),
                                   varBins, minvar, maxvar );
      inVarHTrue[*iIn]->GetXaxis()->SetTitle( (trackType+" "+particleType+" "+getVarAxisLabel(*iIn)).c_str() );
      inVarHFake[*iIn] = new TH1F( ("hFake_" + (*iIn)).c_str(),
                                   ( (*iIn) + " Fake " + particleType +
                                     " " + trackType + " | " + extraHistoInfo ).c_str(),
                                   varBins, minvar, maxvar );
      inVarHFake[*iIn]->GetXaxis()->SetTitle( (trackType+" "+particleType+" "+getVarAxisLabel(*iIn)).c_str() );
      // Eff and misid for MVA cuts
      for ( std::vector<double>::const_iterator iCut = mvaOutCuts.begin();
            iCut != mvaOutCuts.end(); ++iCut )
      {
        histoFile.cd(histoFileRoot.c_str());
        std::ostringstream text;
        text << boost::format("%6.3f") % *iCut;
        std::string cut(text.str()), cCut("MVAcut"+text.str());
        removeSpaces(cut);
        removeSpaces(cCut);
        if ( !mvadirMade[cCut] )
        {
          std::cout << getTime() << "Creating ROOT directory " << cCut << std::endl;
          if ( !histoFile.mkdir(cCut.c_str()) )
          {
            std::cerr << getTime() << " -> Failed to create ROOT dir " << cCut << std::endl;
            return 1;
          }
          mvadirMade[cCut] = true;
        }
        histoFile.cd((histoFileRoot+cCut+"/").c_str());
        TProfile * peff = new TProfile( ("mvaeffVinVar_"+(*iIn)).c_str(),
                                        ( trackType + " " + particleType +
                                          " ID Eff. V " + (*iIn) + " | MVAout > " + cut +
                                          " | " + extraHistoInfo ).c_str(),
                                        varBins, minvar, maxvar );
        peff->GetXaxis()->SetTitle( (trackType+" "+particleType+" "+getVarAxisLabel(*iIn)).c_str() );
        peff->GetYaxis()->SetTitle( "Efficiency / %" );
        (mvaeffVinVar[*iIn])[cCut] = CutProfile( peff, *iCut );
        TProfile * pmisid = new TProfile( ("mvamisidVinVar_"+(*iIn)).c_str(),
                                          ( trackType+" "+particleType +
                                            " Mis-ID Eff. V " + (*iIn) + " | MVAout > " + cut +
                                            " | " + extraHistoInfo ).c_str(),
                                          varBins, minvar, maxvar );
        pmisid->GetXaxis()->SetTitle( (trackType+" "+particleType+" "+getVarAxisLabel(*iIn)).c_str() );
        pmisid->GetYaxis()->SetTitle( "Efficiency / %" );
        (mvamisidVinVar[*iIn])[cCut] = CutProfile( pmisid, *iCut );
      }
      // Eff and misid for DLL cuts
      histoFile.cd(histoFileRoot.c_str());
      for ( std::vector<double>::const_iterator iCut = dllCuts.begin();
            iCut != dllCuts.end(); ++iCut )
      {
        histoFile.cd(histoFileRoot.c_str());
        std::ostringstream text;
        text << boost::format("%6.3f") % *iCut;
        std::string cut(text.str()), cCut("DLLcut"+text.str());
        removeSpaces(cut);
        removeSpaces(cCut);
        if ( !dlldirMade[cCut] )
        {
          std::cout << getTime() << "Creating ROOT directory " << cCut << std::endl;
          if ( !histoFile.mkdir(cCut.c_str()) )
          {
            std::cerr << getTime() << " -> Failed to create ROOT dir " << cCut << std::endl;
            return 1;
          }
          dlldirMade[cCut] = true;
        }
        histoFile.cd((histoFileRoot+cCut+"/").c_str());
        TProfile * eff = new TProfile( ("dlleffVinVar_" + (*iIn)).c_str(),
                                       ( trackType+" "+particleType +
                                         " ID Eff. V " + (*iIn) + " | "+combDllVar+" > " + cut +
                                         " | " + extraHistoInfo ).c_str(),
                                       varBins, minvar, maxvar );
        eff->GetXaxis()->SetTitle( (trackType+" "+particleType+" "+getVarAxisLabel(*iIn)).c_str() );
        eff->GetYaxis()->SetTitle( "Efficiency / %" );
        (dlleffVinVar[*iIn])[cCut] = CutProfile( eff, *iCut );
        TProfile * mis = new TProfile( ("dllmisidVinVar_" + (*iIn)).c_str(),
                                       ( trackType+" "+particleType +
                                         " Mis-ID Eff. V " + (*iIn) + " | "+combDllVar+" > " + cut +
                                         " | " + extraHistoInfo ).c_str(),
                                       varBins, minvar, maxvar );
        eff->GetXaxis()->SetTitle( (trackType+" "+particleType+" "+getVarAxisLabel(*iIn)).c_str() );
        eff->GetYaxis()->SetTitle( "Efficiency / %" );
        (dllmisidVinVar[*iIn])[cCut] = CutProfile( mis, *iCut );
      }
    } // loop over inputs

    histoFile.cd(histoFileRoot.c_str());

    // network output histos
    TH1F * trueMVA = new TH1F( ("hTrue_MVAout"+particleType).c_str(),
                               ( "MVA Output True " + particleType + " " +
                                 trackType + " | " + extraHistoInfo ).c_str(),
                               nBins, mvaMin, mvaMax );
    trueMVA->GetXaxis()->SetTitle( (trackType+" "+particleType+" MVA Output").c_str() );
    TH1F * fakeMVA = new TH1F( ("hFake_MVAout"+particleType).c_str(),
                               ( "MVA Output Fake " + particleType + " " +
                                 trackType + " | " + extraHistoInfo ).c_str(),
                               nBins, mvaMin, mvaMax );
    fakeMVA->GetXaxis()->SetTitle( (trackType+" "+particleType+" MVA Output").c_str() );

    // DLL histos
    TH1F * trueDLL = new TH1F( ("hTrue_DLL"+particleType).c_str(),
                               ( combDllVar + " True " + particleType + " " +
                                 trackType + " | " + extraHistoInfo ).c_str(),
                               nBins, inputMaxMin[combDllVar].min, inputMaxMin[combDllVar].max );
    trueDLL->GetXaxis()->SetTitle( (trackType+" "+particleType+" "+combDllVar).c_str() );
    TH1F * fakeDLL = new TH1F( ("hFake_DLL"+particleType).c_str(),
                               ( combDllVar + " Fake " + particleType + " " +
                                 trackType + " | " + extraHistoInfo ).c_str(),
                               nBins, inputMaxMin[combDllVar].min, inputMaxMin[combDllVar].max );
    fakeDLL->GetXaxis()->SetTitle( (trackType+" "+particleType+" "+combDllVar).c_str() );

    // purity plots
    TProfile * purVmvaOut  =
      new TProfile( (particleType+"purVmvaOut").c_str(),
                    (trackType+" "+particleType+" Purity V MVA output | " + extraHistoInfo ).c_str(),
                    nBins,mvaMin,mvaMax );
    purVmvaOut->GetXaxis()->SetTitle( (trackType+" "+particleType+" MVA Output").c_str() );
    purVmvaOut->GetYaxis()->SetTitle( "Purity / %" );
    TProfile * purVdllOut =
      new TProfile( (particleType+"purVdllOut").c_str(),
                    (trackType+" "+particleType+" Purity V "+combDllVar+" | " + extraHistoInfo ).c_str(),
                    nBins,inputMaxMin[combDllVar].min,inputMaxMin[combDllVar].max );
    purVdllOut->GetXaxis()->SetTitle( (trackType+" "+particleType+" "+combDllVar).c_str() );
    purVdllOut->GetYaxis()->SetTitle( "Purity / %" );

    // Correlation plots
    TH2D * sig_mvaVdll =
      new TH2D( (particleType+"mvaOutVDllSig").c_str(),
                (trackType+" True "+particleType+" MVAout V " + combDllVar + " | " + extraHistoInfo ).c_str(),
                nBins2D,inputMaxMin[combDllVar].min,inputMaxMin[combDllVar].max,
                nBins2D,mvaMin,mvaMax );
    sig_mvaVdll->GetXaxis()->SetTitle( (trackType+" "+particleType+" "+combDllVar).c_str() );
    sig_mvaVdll->GetYaxis()->SetTitle( (trackType+" "+particleType+" MVA Output").c_str() );
    TH2D * bck_mvaVdll =
      new TH2D( (particleType+"mvaOutVDllBck").c_str(),
                (trackType+" Fake "+particleType+" MVAout V " + combDllVar + " | " + extraHistoInfo ).c_str(),
                nBins2D,inputMaxMin[combDllVar].min,inputMaxMin[combDllVar].max,
                nBins2D,mvaMin,mvaMax );
    bck_mvaVdll->GetXaxis()->SetTitle( (trackType+" "+particleType+" "+combDllVar).c_str() );
    bck_mvaVdll->GetYaxis()->SetTitle( (trackType+" "+particleType+" MVA Output").c_str() );

    typedef std::map< double , std::pair<TProfile*,TProfile*> > CutToEffProfiles;

    // MVA Efficiency for DLL selected samples
    histoFile.cd(histoFileRoot.c_str());
    CutToEffProfiles dllCutMVAEff;
    for ( std::vector<double>::const_iterator iCut = dllCuts.begin();
          iCut != dllCuts.end(); ++iCut )
    {
      histoFile.cd(histoFileRoot.c_str());
      std::ostringstream text;
      text << boost::format("%6.3f") % *iCut;
      std::string cut(text.str()), cCut("DLLcut"+text.str());
      removeSpaces(cut);
      removeSpaces(cCut);
      if ( !dlldirMade[cCut] )
      {
        std::cout << getTime() << "Creating ROOT directory " << cCut << std::endl;
        if ( !histoFile.mkdir(cCut.c_str()) )
        {
          std::cerr << getTime() << " -> Failed to create ROOT dir " << cCut << std::endl;
          return 1;
        }
        dlldirMade[cCut] = true;
      }
      histoFile.cd((histoFileRoot+cCut+"/").c_str());
      TProfile * pS = new TProfile( ("MVASignalEffForDLLCut"+cut).c_str(),
                                    ( trackType + " " + particleType +
                                      " Signal Eff V MVAcut | " + combDllVar + " > " + cut +
                                      " | " + extraHistoInfo ).c_str(),
                                    nBins, mvaMin, mvaMax );
      pS->GetXaxis()->SetTitle( (trackType+" "+particleType+" MVA cut value").c_str() );
      pS->GetYaxis()->SetTitle( "Efficiency / %" );
      TProfile * pB = new TProfile( ("MVAFakeEffForDLLCut"+cut).c_str(),
                                    ( trackType + " " + particleType +
                                      " Fake Eff V MVAcut | " + combDllVar + " > " + cut +
                                      " | " + extraHistoInfo ).c_str(),
                                    nBins, mvaMin, mvaMax );
      pB->GetXaxis()->SetTitle( (trackType+" "+particleType+" MVA cut value").c_str() );
      pB->GetYaxis()->SetTitle( "Efficiency / %" );
      dllCutMVAEff[*iCut] = std::make_pair(pS,pB);
    }

    // DLL Efficiency for MVA selected samples
    histoFile.cd(histoFileRoot.c_str());
    CutToEffProfiles mvaCutDLLEff;
    for ( std::vector<double>::const_iterator iCut = mvaOutCuts.begin();
          iCut != mvaOutCuts.end(); ++iCut )
    {
      histoFile.cd(histoFileRoot.c_str());
      std::ostringstream text;
      text << boost::format("%6.3f") % *iCut;
      std::string cut(text.str()), cCut("MVAcut"+text.str());
      removeSpaces(cut);
      removeSpaces(cCut);
      if ( !mvadirMade[cCut] )
      {
        std::cout << getTime() << "Creating ROOT directory " << cCut << std::endl;
        if ( !histoFile.mkdir(cCut.c_str()) )
        {
          std::cerr << getTime() << " -> Failed to create ROOT dir " << cCut << std::endl;
          return 1;
        }
        mvadirMade[cCut] = true;
      }
      histoFile.cd((histoFileRoot+cCut+"/").c_str());
      TProfile * pS = new TProfile( ("DLLSignalEffForMVACut"+cut).c_str(),
                                    ( trackType + " " + particleType +
                                      " Signal Eff V "+combDllVar+" | MVAout > " + cut +
                                      " | " + extraHistoInfo ).c_str(),
                                    nBins, inputMaxMin[combDllVar].min, inputMaxMin[combDllVar].max );
      pS->GetXaxis()->SetTitle( (trackType+" "+particleType+" "+combDllVar+" cut value").c_str() );
      pS->GetYaxis()->SetTitle( "Efficiency / %" );
      TProfile * pB = new TProfile( ("DLLFakeEffForMVACut"+cut).c_str(),
                                    ( trackType + " " + particleType +
                                      " Fake Eff V "+combDllVar+" | MVAout > " + cut +
                                      " | " + extraHistoInfo ).c_str(),
                                    nBins, inputMaxMin[combDllVar].min, inputMaxMin[combDllVar].max );
      pB->GetXaxis()->SetTitle( (trackType+" "+particleType+" "+combDllVar+" cut value").c_str() );
      pB->GetYaxis()->SetTitle( "Efficiency / %" );
      mvaCutDLLEff[*iCut] = std::make_pair(pS,pB);
    }

    // back to root of output ROOT file
    histoFile.cd(histoFileRoot.c_str());

    // Expert for the trained network
    Expert       * nbExpert   = NULL;
    TMVA::Reader * tmvaReader = NULL;
#ifdef ENABLENB
    if ( "NeuroBayes" == mvaType )
    {
      nbExpert = new Expert(paramFileName.c_str());
    }
#endif
    if ( "TMVA" == mvaType )
    {
      tmvaReader = new TMVA::Reader( "!Color:!Silent" );
      // Add variables
      int i(0);
      for ( std::vector<std::string>::const_iterator iIn = inputs.begin();
            iIn != inputs.end(); ++iIn, ++i )
      {
        tmvaReader->AddVariable( (*iIn).c_str(), &(InputArray[i]) );
      }
      // method
      tmvaReader->BookMVA( tmvaMethod, paramFileName.c_str() );
    }

    // open text data file
    //     const std::string textDataFileName = trackType+"-"+particleType+"-InputOutputData.txt";
    //     std::cout << getTime() << "Creating data text file : " << textDataFileName << std::endl;
    //     std::ofstream textDataFile( textDataFileName.c_str(), std::ios_base::out | std::ios_base::trunc );
    //     if ( !textDataFile.is_open() )
    //     {
    //       std::cerr << getTime() << "ERROR opening text file" << std::endl;
    //       return 1;
    //     }
    //     // First line is just the number of entries
    //     textDataFile << inputs.size() << std::endl;

    // reset the counts
    selectedTracks = 0;

    // MC Track eval selection
    TCut mcTrackEvalSel;
    if ( !mcTrackSel( mcTrackSelEval, mcTrackEvalSel ) ) { return 1; }

    // Rescale the output ?
    const bool rescaleOutput = ( "NeuroBayes" == mvaType ||
                                 ( "TMVA" == mvaType && "BDT" == tmvaMethod )
                                 );
    if ( rescaleOutput )
    { std::cout << getTime() << "Will linearly rescale MVA output to range 0-1" << std::endl; }
    const bool applyCorr = ( "TMVA" == mvaType && "MLP" == tmvaMethod && "CE" != tmvaMLPEstimatorType );
    if ( applyCorr )
    { std::cout << getTime() << "Will apply output scaling" << std::endl; }

    // loop over evaluation files as required
    ++iFileName; // Start on first file not used for training.
    for ( ; iFileName != trainingFiles.end(); ++iFileName )
    {

      // load evaluation root file
      std::cout << getTime() << "Using evaluation file " << *iFileName << std::endl;
      TFile * eval_file = TFile::Open((*iFileName).c_str());
      if ( !eval_file )
      {
        std::cerr << getTime() << "Failed to open input ROOT file " << *iFileName << std::endl;
        return 1;
      }
      TTree * eval_tree = (TTree*)gDirectory->Get(tupleLocation.c_str());
      if ( !eval_tree )
      {
        std::cerr << getTime() << "Failed to open MVA TTree" << std::endl;
        return 1;
      }

      // Apply track selection
      eval_tree->Draw( ">> eval_elist", trackSel && mcTrackEvalSel, "entrylist" );
      TEntryList * eval_tree_elist = (TEntryList*) gDirectory->Get("eval_elist");
      if ( !eval_tree_elist )
      { std::cerr << getTime() << "ERROR : Problem getting selected entries" << std::endl; return 1; }
      std::cout << getTime() << "Selected " << eval_tree_elist->GetN() << " entries from "
                << eval_tree->GetEntries()
                << " ( " << 100.0 * eval_tree_elist->GetN() / eval_tree->GetEntries() << " % )"
                << std::endl;

      // Create a Ntuple reader
      tuple_reader = new NTupleReader( eval_tree, inputs );
      // variables that always should be available, even if not part of network inputs
      for ( std::vector<std::string>::const_iterator iV = alwaysVars.begin();
            iV != alwaysVars.end(); ++iV )
      {
        tuple_reader->addVariable( eval_tree, *iV );
      }

      // MC
      Int_t MCParticleType(0);
      TBranch * b_MCParticleType(NULL);
      eval_tree->SetBranchAddress( "MCParticleType", &MCParticleType, &b_MCParticleType );

      // Loop over entry list
      eval_tree->SetEntryList(eval_tree_elist);
      for ( Long64_t iEntry = 0; iEntry < eval_tree_elist->GetN(); ++iEntry )
      {
        const int entryN = eval_tree->GetEntryNumber(iEntry);
        if ( entryN < 0 ) break;
        eval_tree->GetEntry(entryN);
        // if ( (iEntry+1) % (eval_tree_elist->GetN()/10) == 0 )
        // {
        //   std::cout << getTime() << "Read entry " << (1+iEntry)
        //             << " (" << 100.0*((float)(1+iEntry)/(float)eval_tree_elist->GetN()) << "%)"
        //             << std::endl;
        // }

        // Track selection
        if ( !trackTypeSel(tuple_reader->variable("TrackType")) ) continue;

        // ghost treatment
        if ( 0 == MCParticleType && !keepGhostsEval ) continue;

        // Sample events based on # Rich2 hits
        if ( "ReweightRICH2" == reweightOpt && !applyRich2HitsWeight() ) continue;

        // true or fake target
        const bool target = ( abs(MCParticleType) == iPIDtype );

        // Fill an input array for the expert
        int ivar = 0;
        for ( std::vector<std::string>::const_iterator iIn = inputs.begin();
              iIn != inputs.end(); ++iIn, ++ivar )
        {
          InputArray[ivar] = tuple_reader->variable(*iIn);
        }

        // network output
        int original_stdout(0), original_stderr(0);
        if ( nbExpert )
        {
          original_stdout = dup(fileno(stdout));
          fflush(stdout);
          freopen("/dev/null","w",stdout);
          original_stderr = dup(fileno(stderr));
          fflush(stderr);
          freopen("/dev/null","w",stderr);
        }
        double mvaOut = (
#ifdef ENABLENB
                         nbExpert   ? nbExpert->nb_expert(InputArray)             :
#endif
                         tmvaReader ? tmvaReader->EvaluateMVA(tmvaMethod.c_str()) :
                         0.5
                         );
        if ( rescaleOutput ) { mvaOut = ( 1.0 + mvaOut ) / 2.0 ; }
        if ( applyCorr )
        {
          const double e = 0.002;
          mvaOut = ( 1 +
                     std::sqrt(std::pow(mvaOut,2)+4.0*e) -
                     std::sqrt(std::pow(mvaOut-1.0,2)+4.0*e) ) / 2.0;
        }
        // last double check
        if      ( mvaOut > 1.0 ) { mvaOut = 1.0; }
        else if ( mvaOut < 0.0 ) { mvaOut = 0.0; }
        if ( nbExpert )
        {
          fflush(stdout);
          dup2(original_stdout,fileno(stdout));
          close(original_stdout);
          fflush(stderr);
          dup2(original_stderr,fileno(stderr));
          close(original_stderr);
        }

        //       // Write to text file every now and then, for debugging NB later on ...
        //       const int writeFreq = 100000;
        //       if ( writeFreq > eval_tree_elist->GetN() ||
        //            selectedTracks % (eval_tree_elist->GetN()/writeFreq) == 0 )
        //       {
        //         // First the output
        //         textDataFile << boost::lexical_cast<std::string>(mvaOut) << " ";
        //         // then the inputs
        //         for ( unsigned int iivar = 0; iivar < inputs.size(); ++iivar )
        //         {
        //           textDataFile << boost::lexical_cast<std::string>(InputArray[iivar]) << " ";
        //         }
        //         textDataFile << std::endl;
        //       }

        // Combined DLL for this particle type
        const double combdll = combDLL();

        // MVA output plots
        ( target ? trueMVA : fakeMVA ) -> Fill ( mvaOut  );

        // DLL plots
        ( target ? trueDLL : fakeDLL ) -> Fill ( combdll );

        // MVA V DLL plots
        ( target ? sig_mvaVdll : bck_mvaVdll ) -> Fill( combdll, mvaOut );

        // Loop over input variables again to make plots
        ivar = 0;
        for ( std::vector<std::string>::const_iterator iIn = inputs.begin();
              iIn != inputs.end(); ++iIn, ++ivar )
        {

          // fill input plots
          ( target ? inVarHTrue[*iIn] : inVarHFake[*iIn] ) -> Fill ( InputArray[ivar] );

          // MVA Eff and mis ID versus inputs
          const StringTProfileMap& mvamap = ( target ? mvaeffVinVar[*iIn] : mvamisidVinVar[*iIn] );
          for ( StringTProfileMap::const_iterator iI = mvamap.begin();
                iI != mvamap.end(); ++iI )
          {
            iI->second.profile->Fill( InputArray[ivar], mvaOut > iI->second.cut ? 100.0 : 0.0 );
          }

          // DLL Eff and mis ID versus inputs
          const StringTProfileMap& dllmap = ( target ? dlleffVinVar[*iIn] : dllmisidVinVar[*iIn] );
          for ( StringTProfileMap::const_iterator iI = dllmap.begin();
                iI != dllmap.end(); ++iI )
          {
            iI->second.profile->Fill( InputArray[ivar], combdll > iI->second.cut ? 100.0 : 0.0 );
          }

        } // loop over input variables

        // eff V mis curves MVA
        double mvacut(mvaMin);
        for ( unsigned int iCut = 0; iCut < nCutsMVA; ++iCut, mvacut += mvastep )
        {
          // Get the data for this cut
          PIDData & pidData = pidStepDataMVA[mvacut];
          // get the data for this MC particle type
          SelData & selData = pidData[abs(MCParticleType)];
          // do we pass the cut ?
          const bool selected = ( mvaOut > mvacut );
          // count those that pass
          if ( selected ) { ++(selData.first); }
          // count all
          ++(selData.second);
        }

        // eff V mis curves DLL
        double dllcut(dllMin(particleType));
        for ( unsigned int iCut = 0; iCut < nCutsDLL; ++iCut, dllcut += dllstep )
        {
          // Get the data for this cut
          PIDData & pidData = pidStepDataCombDLL[dllcut];
          // get the data for this MC particle type
          SelData & selData = pidData[abs(MCParticleType)];
          // do we pass the cut ?
          const bool selected = ( combdll > dllcut );
          // count those that pass
          if ( selected ) { ++(selData.first);  }
          // count all
          ++(selData.second);
        }

        // purity plots
        purVmvaOut -> Fill( mvaOut,  target ? 100.0 : 0.0 );
        purVdllOut -> Fill( combdll, target ? 100.0 : 0.0 );

        // MVA Eff v DLL cuts
        for ( CutToEffProfiles::iterator iC = dllCutMVAEff.begin();
              iC != dllCutMVAEff.end(); ++iC )
        {
          if ( combdll > iC->first )
          {
            TProfile * p = ( target ? iC->second.first : iC->second.second );
            // loop over bins
            for ( int iBin = 1; iBin <= p->GetNbinsX(); ++iBin )
            {
              // Bin center value
              const double mvaCut = p->GetBinCenter(iBin);
              // Fill eff plot for each bin
              p->Fill( mvaCut, mvaOut > mvaCut ? 100.0 : 0.0 );
            }
          }
        }

        // DLL Eff v ALL cuts
        for ( CutToEffProfiles::iterator iC = mvaCutDLLEff.begin();
              iC != mvaCutDLLEff.end(); ++iC )
        {
          if ( mvaOut > iC->first )
          {
            TProfile * p = ( target ? iC->second.first : iC->second.second );
            // loop over bins
            for ( int iBin = 1; iBin <= p->GetNbinsX(); ++iBin )
            {
              // Bin center value
              const double dllCut = p->GetBinCenter(iBin);
              // Fill eff plot for each bin
              p->Fill( dllCut, combdll > dllCut ? 100.0 : 0.0 );
            }
          }
        }

        // found enough tracks ?
        if ( ++selectedTracks >= nEvalTracks ) break;

      } // loop over ntuple

      // delete the tuple reader
      delete tuple_reader;
      tuple_reader = NULL;

      // Delete entry list
      eval_tree_elist->Delete();

      // close eval file
      eval_file->Close();

      // found enough tracks ?
      if ( selectedTracks >= nEvalTracks ) break;

    } // loop over file names

    // close the text file
    //textDataFile.close();

    // clean up
    delete nbExpert;
    nbExpert = NULL;

    // Make directory for text files.
    if ( !boost::filesystem::exists(boost::filesystem::path("text")) )
      boost::filesystem::create_directory(boost::filesystem::path("text"));

    // Print purity plots
    {
      TCanvas * pnnc =
        new TCanvas( "Purity V MVA Output",
                     (trackType+" "+particleType+" Purity V MVA Output | "
                      + extraHistoInfo ).c_str(),
                     canvasDims.first, canvasDims.second );
      pnnc->SetGrid();
      purVdllOut->SetStats(0);
      purVdllOut->Draw();
      printCanvas( pnnc, trackType+"_"+particleType+"-PurityVCombDLL" );
      purVmvaOut->SetStats(0);
      purVmvaOut->Draw();
      printCanvas( pnnc, trackType+"_"+particleType+"-PurityVMVAOut" );
    }

    // Print MVA output plots
    {
      TCanvas * ac  =
        new TCanvas( "MVA Output",
                     (trackType+" "+particleType+" MVA Output | "
                      + extraHistoInfo ).c_str(),
                     canvasDims.first, canvasDims.second );
      ac->SetGrid();
      fakeMVA->SetStats(0);
      trueMVA->SetStats(0);
      fakeMVA->SetMarkerColor(kRed);
      fakeMVA->SetLineColor(kRed);
      trueMVA->SetMarkerColor(kBlue);
      trueMVA->SetLineColor(kBlue);
      fakeMVA->SetMarkerStyle(kFullTriangleUp);
      trueMVA->SetMarkerStyle(kFullCircle);
      const bool fakeFirst = ( fakeMVA->GetBinContent(fakeMVA->GetMaximumBin()) >
                               trueMVA->GetBinContent(trueMVA->GetMaximumBin()) );
      TLegend * legend = new TLegend(0.435,0.9,0.565,0.85);
      legend->SetTextSize(0.02);
      legend->SetMargin(0.1);
      legend->AddEntry( trueMVA, ("True " + particleType).c_str(), "p" );
      legend->AddEntry( fakeMVA, ("Fake " + particleType).c_str(), "p" );
      ac->SetLogy(false);
      ( fakeFirst ? fakeMVA : trueMVA ) -> Draw("E");
      ( fakeFirst ? trueMVA : fakeMVA ) -> Draw("E SAME");
      legend->Draw();
      printCanvas( ac, trackType+"_"+particleType+"-MVAOut_Liny" );
      ac->SetLogy(true);
      ( fakeFirst ? fakeMVA : trueMVA ) -> Draw("E");
      ( fakeFirst ? trueMVA : fakeMVA ) -> Draw("E SAME");
      legend->Draw();
      printCanvas( ac, trackType+"_"+particleType+"-MVAOut_Logy" );
    }

    // Print DLL output plots
    {
      TCanvas * ac  =
        new TCanvas( "DLL Output",
                     (trackType+" "+particleType+" DLL Output | "
                      + extraHistoInfo ).c_str(),
                     canvasDims.first, canvasDims.second );
      ac->SetGrid();
      fakeDLL->SetStats(0);
      trueDLL->SetStats(0);
      fakeDLL->SetMarkerColor(kRed);
      fakeDLL->SetLineColor(kRed);
      trueDLL->SetMarkerColor(kBlue);
      trueDLL->SetLineColor(kBlue);
      fakeDLL->SetMarkerStyle(kFullTriangleUp);
      trueDLL->SetMarkerStyle(kFullCircle);
      const bool fakeFirst = ( fakeDLL->GetBinContent(fakeDLL->GetMaximumBin()) >
                               trueDLL->GetBinContent(trueDLL->GetMaximumBin()) );
      TLegend * legend = new TLegend(0.435,0.9,0.565,0.85);
      legend->SetTextSize(0.02);
      legend->SetMargin(0.1);
      legend->AddEntry( trueDLL, ("True " + particleType).c_str(), "p" );
      legend->AddEntry( fakeDLL, ("Fake " + particleType).c_str(), "p" );
      ac->SetLogy(false);
      ( fakeFirst ? fakeDLL : trueDLL ) -> Draw("E");
      ( fakeFirst ? trueDLL : fakeDLL ) -> Draw("E SAME");
      legend->Draw();
      printCanvas( ac, trackType+"_"+particleType+"-"+combDllVar+"_Liny" );
      ac->SetLogy(true);
      ( fakeFirst ? fakeDLL : trueDLL ) -> Draw("E");
      ( fakeFirst ? trueDLL : fakeDLL ) -> Draw("E SAME");
      legend->Draw();
      printCanvas( ac, trackType+"_"+particleType+"-"+combDllVar+"_Logy" );
    }

    // Correlation plots
    {
      TCanvas * ac =
        new TCanvas( "MVA Output V DLL",
                     (trackType+" "+particleType+" MVA Output V DLL | "
                      + extraHistoInfo ).c_str(),
                     canvasDims.first, canvasDims.second );
      ac->SetGrid();
      sig_mvaVdll->SetStats(0);
      bck_mvaVdll->SetStats(0);
      ac->SetLogz(false);
      sig_mvaVdll->Draw("zcol");
      printCanvas( ac, trackType+"_"+particleType+"-MVAOutVDLL-Signal_Linz" );
      bck_mvaVdll->Draw("zcol");
      printCanvas( ac, trackType+"_"+particleType+"-MVAOutVDLL-Background_Linz" );
      ac->SetLogz(true);
      sig_mvaVdll->Draw("zcol");
      printCanvas( ac, trackType+"_"+particleType+"-MVAOutVDLL-Signal_Logz" );
      bck_mvaVdll->Draw("zcol");
      printCanvas( ac, trackType+"_"+particleType+"-MVAOutVDLL-Background_Logz" );
    }

    // MVA Eff Eff V momentum
    makeEffVVarPlots( "MVAEffVTrackP",  mvaeffVinVar["TrackP"],  mvamisidVinVar["TrackP"]  );
    makeEffVVarPlots( "MVAEffVTrackPt", mvaeffVinVar["TrackPt"], mvamisidVinVar["TrackPt"] );
    makeEffVVarPlots( "DLLEffVTrackP",  dlleffVinVar["TrackP"],  dllmisidVinVar["TrackP"]  );
    makeEffVVarPlots( "DLLEffVTrackPt", dlleffVinVar["TrackPt"], dllmisidVinVar["TrackPt"] );

    // MVA Eff for DLL cut plots
    {
      const std::string effDir("MVAEffForDLLCut");
      TCanvas * c =
        new TCanvas( "MVA Eff for DLL Cut",
                     (trackType+" "+particleType+" MVA Eff for DLL Cut | "
                      + extraHistoInfo ).c_str(),
                     canvasDims.first, canvasDims.second );
      c->SetLogy(false);
      c->SetGrid();
      for ( CutToEffProfiles::iterator iC = dllCutMVAEff.begin();
            iC != dllCutMVAEff.end(); ++iC )
      {
        TProfile * pS = iC->second.first;
        TProfile * pB = iC->second.second;
        if ( pS && pB )
        {
          std::ostringstream text;
          text << boost::format("%6.3f") % iC->first;
          std::string cut = text.str();
          removeSpaces(cut);
          pB->SetStats(0);
          pS->SetStats(0);
          pB->SetMarkerColor(kRed);
          pB->SetLineColor(kRed);
          pS->SetMarkerColor(kBlue);
          pS->SetLineColor(kBlue);
          TLegend * legend = new TLegend(0.8,0.9,0.9,0.85);
          legend->SetTextSize(0.02);
          legend->SetMargin(0.1);
          legend->AddEntry( pS, ("True " + particleType).c_str(), "p" );
          legend->AddEntry( pB, ("Fake " + particleType).c_str(), "p" );
          pS -> Draw("E");
          pB -> Draw("E SAME");
          legend->Draw();
          printCanvas( c, effDir+"/"+trackType+"_"+particleType+"-MVAEff-DLLCut"+cut );
        }
      }
    }

    // DLL Eff for MVA cut plots
    {
      const std::string effDir("DLLEffForMVACut");
      TCanvas * c =
        new TCanvas( "DLL Eff for MVA Cut",
                     (trackType+" "+particleType+" DLL Eff for MVA Cut | "
                      + extraHistoInfo ).c_str(),
                     canvasDims.first, canvasDims.second );
      c->SetLogy(false);
      c->SetGrid();
      for ( CutToEffProfiles::iterator iC = mvaCutDLLEff.begin();
            iC != mvaCutDLLEff.end(); ++iC )
      {
        TProfile * pS = iC->second.first;
        TProfile * pB = iC->second.second;
        if ( pS && pB )
        {
          std::ostringstream text;
          text << boost::format("%6.3f") % iC->first;
          std::string cut = text.str();
          removeSpaces(cut);
          pB->SetStats(0);
          pS->SetStats(0);
          pB->SetMarkerColor(kRed);
          pB->SetLineColor(kRed);
          pS->SetMarkerColor(kBlue);
          pS->SetLineColor(kBlue);
          TLegend * legend = new TLegend(0.8,0.9,0.9,0.85);
          legend->SetTextSize(0.02);
          legend->SetMargin(0.1);
          legend->AddEntry( pS, ("True " + particleType).c_str(), "p" );
          legend->AddEntry( pB, ("Fake " + particleType).c_str(), "p" );
          pS -> Draw("E");
          pB -> Draw("E SAME");
          legend->Draw();
          printCanvas( c, effDir+"/"+trackType+"_"+particleType+"-DLLEff-MVACut"+cut );
        }
      }
    }

    // Loop over the cut data and make various eff V pur cut plots
    std::cout << getTime() << "Making final performance plots ..." << std::endl;
    for ( StringToInt::const_iterator iH = particlePDGcodes().begin();
          iH != particlePDGcodes().end(); ++iH )
    {
      const std::string& iMisName = iH->first;
      const int iMisID            = iH->second;
      if ( iMisID == iPIDtype ) continue;

      const double minMisIDeff(0.05),maxMisIDeff(80),minIDeff(30),maxIDeff(110);
      double minPur(0), maxPur(0), minEffForPur(0), maxEffForPur(0); // set by call below
      getMinMaxEffPur(trackType,particleType,minPur,maxPur,minEffForPur,maxEffForPur);

      // MVA Plots
      std::vector<double> mva_ideff, mva_idefferr, mva_misideff, mva_misidefferr;
      std::vector<double> mva_ideffForP, mva_idefferrForP, mva_purity, mva_purityerr;
      std::vector<double> mva_ideffForPP, mva_idefferrForPP;
      std::vector<double> mva_allMisID, mva_allMisIDErr;
      PlotLabels mva_plotLabels, mva_plotLabelsForP, mva_plotLabelsAllMisID;
      unsigned int mva_nCut(0);
      const unsigned int mva_labelInc(5);
      {
        for ( PIDStepData::iterator iStep = pidStepDataMVA.begin();
              iStep != pidStepDataMVA.end(); ++iStep, ++mva_nCut )
        {
          // Cut value
          const double cut = iStep->first;

          // ID
          const double selPart    = ((iStep->second)[iPIDtype]).first;
          const double totPart    = ((iStep->second)[iPIDtype]).second;
          const double partEff    = 100. * getEff( selPart, totPart );
          const double partEffErr = 100. * poisError( selPart, totPart );

          // misID
          const double selPartMisID    = ((iStep->second)[iMisID]).first;
          const double totPartMisID    = ((iStep->second)[iMisID]).second;
          const double partMisIDEff    = 100. * getEff( selPartMisID, totPartMisID );
          const double partMisIDEffErr = 100. * poisError( selPartMisID, totPartMisID );

          // fill vectors
          if ( partEff      >= minIDeff    &&
               partEff      <= maxIDeff    &&
               partMisIDEff >= minMisIDeff &&
               partMisIDEff <= maxMisIDeff  )
          {
            // fill data vectors
            mva_ideff.push_back       ( partEff         );
            mva_idefferr.push_back    ( partEffErr      );
            mva_misideff.push_back    ( partMisIDEff    );
            mva_misidefferr.push_back ( partMisIDEffErr );
            // add to list of plot labels
            if ( mva_nCut % mva_labelInc == 0 )
            {
              mva_plotLabels.push_back( PlotLabel(cut,1.0+partEff,partMisIDEff) );
            }
          }

          // loop over all PID types and count all particles selected for this cut
          unsigned int totalSelPart(0), totNonTarget(0), selNonTarget(0);
          for ( PIDData::const_iterator iMN = (iStep->second).begin();
                iMN != (iStep->second).end(); ++iMN )
          {
            totalSelPart += iMN->second.first;
            if ( iMN->first != (unsigned int)iPIDtype )
            {
              selNonTarget += iMN->second.first;
              totNonTarget += iMN->second.second;
            }
          }

          // purity
          const double purity    = 100.0 * getEff( selPart, totalSelPart );
          const double purityErr = 100.0 * poisError( purity/100.0, totalSelPart );

          // overall mis-ID prob
          const double allMisID    = 100.0 * getEff( selNonTarget, totNonTarget );
          const double allMisIDErr = 100.0 * poisError( allMisID/100.0, totNonTarget );

          // fill vectors
          if ( partEff      >= minEffForPur &&
               partEff      <= maxEffForPur &&
               purity       >= minPur       &&
               purity       <= maxPur        )
          {
            mva_ideffForP.push_back    ( partEff    );
            mva_idefferrForP.push_back ( partEffErr );
            mva_purity.push_back       ( purity     );
            mva_purityerr.push_back    ( purityErr  );
            // add to list of plot labels
            if ( mva_nCut % mva_labelInc == 0 )
            {
              mva_plotLabelsForP.push_back( PlotLabel(cut,1.0+partEff,purity) );
            }
          }
          if ( partEff  >= minIDeff    &&
               partEff  <= maxIDeff    &&
               allMisID >= minMisIDeff &&
               allMisID <= maxMisIDeff  )
          {
            mva_ideffForPP.push_back    ( partEff    );
            mva_idefferrForPP.push_back ( partEffErr );
            mva_allMisID.push_back      ( allMisID   );
            mva_allMisIDErr.push_back   ( allMisIDErr );
            // add to list of plot labels
            if ( mva_nCut % mva_labelInc == 0 )
            {
              mva_plotLabelsAllMisID.push_back( PlotLabel(cut,1.0+partEff,allMisID) );
            }
          }

        }
      }

      // DLL plots
      std::vector<double> dll_ideff, dll_idefferr, dll_misideff, dll_misidefferr;
      std::vector<double> dll_ideffForP, dll_idefferrForP, dll_purity, dll_purityerr;
      std::vector<double> dll_ideffForPP, dll_idefferrForPP;
      std::vector<double> dll_allMisID, dll_allMisIDErr;
      PlotLabels dll_plotLabels, dll_plotLabelsForP, dll_plotLabelsAllMisID;
      unsigned int dll_nCut(0);
      const unsigned int dll_labelInc(5);
      {
        for ( PIDStepData::iterator iStep = pidStepDataCombDLL.begin();
              iStep != pidStepDataCombDLL.end(); ++iStep, ++dll_nCut )
        {
          // Cut value
          const double cut = iStep->first;

          // ID
          const double selPart    = ((iStep->second)[iPIDtype]).first;
          const double totPart    = ((iStep->second)[iPIDtype]).second;
          const double partEff    = 100. * getEff( selPart, totPart );
          const double partEffErr = 100. * poisError( selPart, totPart );

          // misID
          const double selPartMisID    = ((iStep->second)[iMisID]).first;
          const double totPartMisID    = ((iStep->second)[iMisID]).second;
          const double partMisIDEff    = 100. * getEff( selPartMisID, totPartMisID );
          const double partMisIDEffErr = 100. * poisError( selPartMisID, totPartMisID );

          if ( partEff      >= minIDeff    &&
               partEff      <= maxIDeff    &&
               partMisIDEff >= minMisIDeff &&
               partMisIDEff <= maxMisIDeff  )
          {
            // fill data vectors
            dll_ideff.push_back       ( partEff         );
            dll_idefferr.push_back    ( partEffErr      );
            dll_misideff.push_back    ( partMisIDEff    );
            dll_misidefferr.push_back ( partMisIDEffErr );
            // add to list of plot labels
            if ( 0 == (dll_nCut%dll_labelInc) )
            {
              std::ostringstream text;
              text << boost::format("%6.2f") % cut;
              dll_plotLabels.push_back( PlotLabel(text.str(),1.0+partEff,partMisIDEff) );
            }
          }

          // loop over all PID types and count all particles selected for this cut
          unsigned int totalAllPart(0), totNonTarget(0), selNonTarget(0);
          for ( PIDData::const_iterator iMN = (iStep->second).begin();
                iMN != (iStep->second).end(); ++iMN )
          {
            totalAllPart += iMN->second.first;
            if ( iMN->first != (unsigned int)iPIDtype )
            {
              selNonTarget += iMN->second.first;
              totNonTarget += iMN->second.second;
            }
          }

          // purity
          const double purity    = 100. * getEff( selPart, totalAllPart );
          const double purityErr = 100. * poisError( purity/100.0, totalAllPart );

          // overall mis-ID prob
          const double allMisID    = 100.0 * getEff( selNonTarget, totNonTarget );
          const double allMisIDErr = 100.0 * poisError( allMisID/100.0, totNonTarget );

          // fill vectors
          if ( partEff      >= minEffForPur &&
               partEff      <= maxEffForPur &&
               purity       >= minPur       &&
               purity       <= maxPur        )
          {
            dll_ideffForP.push_back    ( partEff    );
            dll_idefferrForP.push_back ( partEffErr );
            dll_purity.push_back       ( purity     );
            dll_purityerr.push_back    ( purityErr  );
            // add to list of plot labels
            if ( dll_nCut % dll_labelInc == 0 )
            {
              std::ostringstream text;
              text << boost::format("%6.2f") % cut;
              dll_plotLabelsForP.push_back( PlotLabel(text.str(),1.0+partEff,purity) );
            }
          }
          if ( partEff  >= minIDeff    &&
               partEff  <= maxIDeff    &&
               allMisID >= minMisIDeff &&
               allMisID <= maxMisIDeff  )
          {
            dll_ideffForPP.push_back    ( partEff    );
            dll_idefferrForPP.push_back ( partEffErr );
            dll_allMisID.push_back      ( allMisID   );
            dll_allMisIDErr.push_back   ( allMisIDErr );
            // add to list of plot labels
            if ( dll_nCut % dll_labelInc == 0 )
            {
              std::ostringstream text;
              text << boost::format("%6.2f") % cut;
              dll_plotLabelsAllMisID.push_back( PlotLabel(text.str(),1.0+partEff,allMisID) );
            }
          }

        }
      }

      if ( !mva_ideff.empty() && !dll_ideff.empty()    &&
           ( mva_ideff.size() == mva_misideff.size() ) &&
           ( mva_ideff.size() == mva_idefferr.size() ) &&
           ( mva_ideff.size() == mva_misidefferr.size() ) &&
           ( dll_ideff.size() == dll_misideff.size() ) )
      {

        TCanvas * c =
          new TCanvas( (iMisName+"cPurEff").c_str(),
                       (trackType+" "+particleType+" ID Eff. V "+iMisName+" MisID Eff. | "
                        + extraHistoInfo ).c_str(),
                       canvasDims.first, canvasDims.second );
        c->SetLogy(true);
        c->SetGrid();

        TGraphErrors * gA = new TGraphErrors( mva_ideff.size(),
                                              &*mva_ideff.begin(),
                                              &*mva_misideff.begin(),
                                              &*mva_idefferr.begin(),
                                              &*mva_misidefferr.begin() );
        gA->SetMarkerColor(kBlue);
        gA->SetLineColor(kBlue);
        gA->SetMarkerStyle(kFullCircle);
        gA->SetTitle( (trackType+" "+particleType+ " ID Eff. V "+iMisName+" MisID Eff. | "
                       + extraHistoInfo ).c_str() );
        gA->GetXaxis()->SetTitle( (particleType+" ID Efficiency / %").c_str() );
        gA->GetYaxis()->SetTitle( (iMisName+" MisID Efficiency / %").c_str() );
        setGraphLimits( gA, minIDeff, maxIDeff, minMisIDeff, maxMisIDeff );
        gA->Draw("ALP");

        TGraphErrors * gB = new TGraphErrors( dll_ideff.size(),
                                              &*dll_ideff.begin(),
                                              &*dll_misideff.begin(),
                                              &*dll_idefferr.begin(),
                                              &*dll_misidefferr.begin() );
        gB->SetMarkerColor(kRed);
        gB->SetLineColor(kRed);
        gB->SetMarkerStyle(kFullTriangleUp);
        gB->Draw("LP");
        // cut point labels
        addCutValues( mva_plotLabels, dll_plotLabels );

        // MVA types
        TLegend * l = new TLegend(0.1,0.9,0.3,0.85);
        l->SetTextSize(0.02);
        l->SetMargin(0.1);
        l->AddEntry( gA, (particleType+" MVA PID").c_str(), "p" );
        l->AddEntry( gB, combDllVar.c_str(), "p" );
        l->Draw();

        c->Update();
        printCanvas( c, trackType+"_"+particleType+"-IDEff_V_"+iMisName+"-MisIDEff" );

        // Dump Eff V MisID data to plain text file
        const std::string textDataFileName = "text/"+trackType+particleType+"EffV"+iMisName+"MisID-"+mvaTextForFile+".txt";
        std::cout << getTime() << "Creating Eff V " << iMisName << " MisID text file : " << textDataFileName << std::endl;
        std::ofstream textDataFile( textDataFileName.c_str(), std::ios_base::out | std::ios_base::trunc );
        textDataFile << mvaTextForFile << std::endl;
        textDataFile << trackType << " " << particleType << std::endl;
        textDataFile << particleType << " ID Efficiency / %" << std::endl;
        textDataFile << iMisName << " MisID Efficiency / %" << std::endl;
        textDataFile << minIDeff << " "<< maxIDeff << " " << minMisIDeff << " " << maxMisIDeff << std::endl;
        for ( unsigned int i = 0; i < mva_ideff.size(); ++i )
        {
          textDataFile << checkValue(mva_ideff[i])    << " " << checkValue(mva_idefferr[i]) << " "
                       << checkValue(mva_misideff[i]) << " " << checkValue(mva_misidefferr[i])
                       << std::endl;
        }
        textDataFile.close();

      }

      static bool madeEffPur(false);
      if ( !madeEffPur &&
           !mva_ideffForP.empty() && !dll_ideffForP.empty() &&
           ( mva_ideffForP.size() == mva_purity.size() )   &&
           ( dll_ideffForP.size() == dll_purity.size() )    )
      {
        madeEffPur = true;

        TCanvas * c  =
          new TCanvas( (particleType+"EffVPurity").c_str(),
                       (trackType + " " + particleType +
                        " Pur. V Eff. | " + extraHistoInfo ).c_str(),
                       canvasDims.first, canvasDims.second );
        c->SetLogy(false);
        c->SetGrid();

        TGraphErrors * gC = new TGraphErrors( mva_ideffForP.size(),
                                              &*mva_ideffForP.begin(),
                                              &*mva_purity.begin(),
                                              &*mva_idefferrForP.begin(),
                                              &*mva_purityerr.begin() );
        gC->SetMarkerColor(kBlue);
        gC->SetLineColor(kBlue);
        gC->SetMarkerStyle(kFullCircle);
        gC->SetTitle( (trackType + " " + particleType +
                       " ID Eff. V Purity | " + extraHistoInfo ).c_str() );
        gC->GetXaxis()->SetTitle( (particleType+" ID Efficiency / %").c_str() );
        gC->GetYaxis()->SetTitle( (particleType+" Purity / %").c_str() );
        setGraphLimits( gC, minEffForPur, maxEffForPur, minPur, maxPur );
        gC->Draw("ALP");

        TGraphErrors * gD = new TGraphErrors( dll_ideffForP.size(),
                                              &*dll_ideffForP.begin(),
                                              &*dll_purity.begin(),
                                              &*dll_idefferrForP.begin(),
                                              &*dll_purityerr.begin() );
        gD->SetMarkerColor(kRed);
        gD->SetLineColor(kRed);
        gD->SetMarkerStyle(kFullTriangleUp);
        gD->Draw("LP");

        // cut point labels
        addCutValues( mva_plotLabelsForP, dll_plotLabelsForP );

        // legion
        TLegend * ll = new TLegend(0.9,0.9,0.7,0.85);
        ll->SetTextSize(0.02);
        ll->SetMargin(0.1);
        ll->AddEntry( gC, (particleType+" MVA PID").c_str(), "p" );
        ll->AddEntry( gD, combDllVar.c_str(), "p" );
        ll->Draw();

        c->Update();
        printCanvas( c, trackType+"_"+particleType+"-IDEff_V_"+particleType+"-Purity" );

        // Dump Eff V Purity data to plain text file
        const std::string textDataFileName = "text/"+trackType+particleType+"EffVPurity-"+mvaTextForFile+".txt";
        std::cout << getTime() << "Creating Eff V Purity text file : " << textDataFileName << std::endl;
        std::ofstream textDataFile( textDataFileName.c_str(), std::ios_base::out | std::ios_base::trunc );
        textDataFile << mvaTextForFile << std::endl;
        textDataFile << trackType << " " << particleType << std::endl;
        textDataFile << particleType << " ID Efficiency / %" << std::endl;
        textDataFile << particleType << " Purity / %" << std::endl;
        textDataFile << minEffForPur << " " << maxEffForPur << " " << minPur << " " << maxPur << std::endl;
        for ( unsigned int i = 0; i < mva_ideffForP.size(); ++i )
        {
          textDataFile << checkValue(mva_ideffForP[i]) << " " << checkValue(mva_idefferrForP[i]) << " "
                       << checkValue(mva_purity[i])    << " " << checkValue(mva_purityerr[i])
                       << std::endl;
        }
        textDataFile.close();

      } // make eff v pur plot

      static bool madeEffAllMisID(false);
      if ( !madeEffAllMisID &&
           !mva_ideffForPP.empty() && !dll_ideffForPP.empty() &&
           ( mva_ideffForPP.size() == mva_allMisID.size() ) &&
           ( dll_ideffForPP.size() == dll_allMisID.size() ) )
      {
        madeEffAllMisID = true;

        TGraphErrors * gC = new TGraphErrors( mva_ideffForPP.size(),
                                              &*mva_ideffForPP.begin(),
                                              &*mva_allMisID.begin(),
                                              &*mva_idefferrForPP.begin(),
                                              &*mva_allMisIDErr.begin() );
        gC->SetMarkerColor(kBlue);
        gC->SetLineColor(kBlue);
        gC->SetMarkerStyle(kFullCircle);
        gC->SetTitle( (trackType+" "+particleType+" ID Eff. V Overall Mis-ID | "
                       + extraHistoInfo ).c_str() );
        const std::string xTitle = particleType+" ID Efficiency / %";
        const std::string yTitle = "Overall misID Efficiency / %";
        gC->GetXaxis()->SetTitle( xTitle.c_str() );
        gC->GetYaxis()->SetTitle( yTitle.c_str() );
        setGraphLimits( gC, minIDeff, maxIDeff, minMisIDeff, maxMisIDeff );

        TGraphErrors * gD = new TGraphErrors( dll_ideffForPP.size(),
                                              &*dll_ideffForPP.begin(),
                                              &*dll_allMisID.begin(),
                                              &*dll_idefferrForPP.begin(),
                                              &*dll_allMisIDErr.begin() );
        gD->SetMarkerColor(kRed);
        gD->SetLineColor(kRed);
        gD->SetMarkerStyle(kFullTriangleUp);

        // Define the legion
        TLegend * ll = new TLegend(0.1,0.9,0.3,0.85);
        ll->SetTextSize(0.02);
        ll->SetMargin(0.1);
        ll->AddEntry( gC, (particleType+" MVA PID").c_str(), "p" );
        ll->AddEntry( gD, combDllVar.c_str(), "p" );

        TCanvas * c  =
          new TCanvas( (particleType+"EffVOverallMisID").c_str(),
                       (trackType+" "+particleType+" ID Eff. V Overall Mis-ID | " +
                        extraHistoInfo).c_str(),
                       canvasDims.first, canvasDims.second );
        c->SetLogy(false);
        c->SetGrid();

        gC->Draw("ALP");
        gD->Draw("LP");

        // cut point labels
        addCutValues( mva_plotLabelsAllMisID, dll_plotLabelsAllMisID );

        // Draw legion last
        ll->Draw();

        c->Update();
        printCanvas( c, trackType+"_"+particleType+"-IDEff_V_OverallMisID_Liny" );

        TCanvas * cc =
          new TCanvas( (particleType+"EffVOverallMisIDLogy").c_str(),
                       (trackType+" "+particleType+" ID Eff. V Overall Mis-ID | "
                        + extraHistoInfo ).c_str(),
                       canvasDims.first, canvasDims.second );
        cc->SetLogy(true);
        cc->SetGrid();

        gC->GetXaxis()->SetTitle( xTitle.c_str() );
        gC->GetYaxis()->SetTitle( yTitle.c_str() );
        gC->Draw("ALP");
        gD->Draw("LP");

        // cut point labels
        addCutValues( mva_plotLabelsAllMisID, dll_plotLabelsAllMisID );

        // Draw legion last
        ll->Draw();

        cc->Update();
        printCanvas( cc, trackType+"_"+particleType+"-IDEff_V_OverallMisID_Logy" );
        cc->SetLogy(false);

        // Dump Eff V Overall MisID data to plain text file
        const std::string textDataFileName = "text/"+trackType+particleType+"EffVOverallMisID-"+mvaTextForFile+".txt";
        std::cout << getTime() << "Creating Eff V OverallMisID text file : " << textDataFileName << std::endl;
        std::ofstream textDataFile( textDataFileName.c_str(), std::ios_base::out | std::ios_base::trunc );
        textDataFile << mvaTextForFile << std::endl;
        textDataFile << trackType << " " << particleType << std::endl;
        textDataFile << particleType << " ID Efficiency / %" << std::endl;
        textDataFile << "Overall MisID Efficiency / %" << std::endl;
        textDataFile << minIDeff << " "<< maxIDeff << " " << minMisIDeff << " " << maxMisIDeff << std::endl;
        for ( unsigned int i = 0; i < mva_ideffForPP.size(); ++i )
        {
          textDataFile << checkValue(mva_ideffForPP[i]) << " " << checkValue(mva_idefferrForPP[i]) << " "
                       << checkValue(mva_allMisID[i])   << " " << checkValue(mva_allMisIDErr[i])
                       << std::endl;
        }
        textDataFile.close();

      } // make eff v all mis ID plot

    } // loop over particle types

    // close histogram file
    histoFile.Write();
    histoFile.Close();

    std::cout << getTime() << "Finished..." << std::endl;

  }

  return 0;
}
