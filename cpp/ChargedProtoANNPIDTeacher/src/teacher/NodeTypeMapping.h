// $Id: $
#ifndef NODETYPEMAPPING_H
#define NODETYPEMAPPING_H 1

#include <map>
#include <string>

// boost
#include "boost/assign/list_of.hpp"

static const int RegSplineFit           = 14;
static const int RegSplineFitAndDelta   = 34;
static const int UnorderedClass         = 18;
static const int OrderedClass           = 19;

typedef std::map<std::string,int> NeuroBayesNodeTypeMap;
static const NeuroBayesNodeTypeMap s_NBnodeTypeMap = boost::assign::map_list_of

  // Event variables
  //   ( "NumLongTracks",        OrderedClass )
  //   ( "NumDownstreamTracks",  OrderedClass )
  //   ( "NumUpstreamTracks",    OrderedClass )
  //   ( "NumVeloTracks",        OrderedClass )
  //   ( "NumTTracks",           OrderedClass )
  //   ( "NumPVs",               OrderedClass )
  //   ( "NumMuonTracks",        RegSplineFit )
  //   ( "NumCaloHypos",         OrderedClass )
  ( "NumProtoParticles",    RegSplineFit )
  ( "NumSPDHits",           RegSplineFit )
  ( "NumRich1Hits",         RegSplineFit )
  ( "NumRich2Hits",         RegSplineFit )

  // Tracking variables
  ( "TrackP",            RegSplineFitAndDelta )
  ( "TrackPt",           RegSplineFitAndDelta )
  ( "TrackChi2PerDof",   RegSplineFit )
  ( "TrackType",         UnorderedClass )
  ( "TrackHistory",      UnorderedClass )
  ( "TrackNumDof",       OrderedClass   )
  ( "TrackLikelihood",   RegSplineFitAndDelta )
  ( "TrackGhostProbability", RegSplineFitAndDelta )
  ( "TrackFitMatchChi2", RegSplineFitAndDelta )
  ( "TrackCloneDist",    RegSplineFitAndDelta )
  ( "TrackFitVeloChi2",  RegSplineFit )
  ( "TrackFitVeloNDoF",  OrderedClass )
  ( "TrackFitTChi2",     RegSplineFit )
  ( "TrackFitTNDoF",     OrderedClass )

  // RICH Variables
  ( "RichUsedAero",      UnorderedClass )
  ( "RichUsedR1Gas",     UnorderedClass )
  ( "RichUsedR2Gas",     UnorderedClass )
  ( "RichAboveElThres",  UnorderedClass )
  ( "RichAboveMuThres",  UnorderedClass )
  ( "RichAbovePiThres",  UnorderedClass )
  ( "RichAboveKaThres",  UnorderedClass )
  ( "RichAbovePrThres",  UnorderedClass )
  ( "RichDLLe",          RegSplineFitAndDelta )
  ( "RichDLLmu",         RegSplineFitAndDelta )
  ( "RichDLLpi",         RegSplineFitAndDelta )
  ( "RichDLLk",          RegSplineFitAndDelta )
  ( "RichDLLp",          RegSplineFitAndDelta )
  ( "RichDLLbt",         RegSplineFitAndDelta )

  // Combined DLL values
  ( "CombDLLe",          RegSplineFitAndDelta )
  ( "CombDLLmu",         RegSplineFitAndDelta )
  ( "CombDLLpi",         RegSplineFitAndDelta )
  ( "CombDLLk",          RegSplineFitAndDelta )
  ( "CombDLLp",          RegSplineFitAndDelta )

  // MUON
  ( "InAccMuon",         UnorderedClass )
  ( "MuonBkgLL",         RegSplineFitAndDelta )
  ( "MuonMuLL",          RegSplineFitAndDelta )
  ( "MuonIsMuon",        UnorderedClass )
  ( "MuonIsLooseMuon",   UnorderedClass )
  ( "MuonNShared",       OrderedClass )

  // ECAL
  ( "InAccEcal",         UnorderedClass )
  ( "CaloChargedSpd",    RegSplineFitAndDelta )
  ( "CaloChargedPrs",    RegSplineFitAndDelta )
  ( "CaloChargedEcal",   RegSplineFit )
  ( "CaloElectronMatch", RegSplineFitAndDelta )
  ( "CaloTrMatch",       RegSplineFitAndDelta )
  ( "CaloEcalE",         RegSplineFit )
  ( "CaloEcalChi2",      RegSplineFitAndDelta )
  ( "CaloClusChi2",      RegSplineFitAndDelta )
  ( "EcalPIDe",          RegSplineFitAndDelta )
  ( "EcalPIDmu",         RegSplineFitAndDelta )
  ( "CaloTrajectoryL",   RegSplineFit )

  // HCAL
  ( "InAccHcal",         UnorderedClass )
  ( "CaloHcalE",         RegSplineFitAndDelta )
  ( "HcalPIDe",          RegSplineFitAndDelta )
  ( "HcalPIDmu",         RegSplineFitAndDelta )

  // PRS
  ( "InAccPrs",          UnorderedClass )
  ( "CaloPrsE",          RegSplineFitAndDelta )
  ( "PrsPIDe",           RegSplineFitAndDelta )

  // SPD
  ( "InAccSpd",          UnorderedClass )
  ( "CaloSpdE",          RegSplineFitAndDelta )

  // BREM
  ( "InAccBrem",         UnorderedClass )
  ( "CaloNeutralSpd",    RegSplineFitAndDelta )
  ( "CaloNeutralPrs",    RegSplineFitAndDelta )
  ( "CaloNeutralEcal",   RegSplineFit )
  ( "CaloBremMatch",     RegSplineFitAndDelta )
  ( "CaloBremChi2",      RegSplineFitAndDelta )
  ( "BremPIDe",          RegSplineFitAndDelta )

  // VELO
  ( "VeloCharge",        RegSplineFitAndDelta )

  // 'Spikes' (Due to CALO delta functions)
  ( "CombDlleSpike",          UnorderedClass )
  ( "CombDllmuSpike",         UnorderedClass )
  ( "CaloElectronMatchSpike", UnorderedClass )
  ( "CaloBremMatchSpike",     UnorderedClass )
  ( "BremPIDeSpike",          UnorderedClass )
  ( "PrsPIDeSpike",           UnorderedClass )
  ( "HcalPIDeSpike",          UnorderedClass )
  ( "EcalPIDeSpike",          UnorderedClass )
  ( "CaloEcalChi2Spike",      UnorderedClass )
  ( "CaloClusChi2Spike",      UnorderedClass )
  ( "CaloPrsESpike",          UnorderedClass )
  ( "CaloBremChi2Spike",      UnorderedClass )
  ;

typedef std::map<std::string,char> TMVANodeTypeMap;
const char& getTMVANodeType( const std::string & var )
{
  static const char defType = 'F';
  static const TMVANodeTypeMap map = boost::assign::map_list_of
    ( "NumProtoParticles", 'I' )
    ( "NumSPDHits",        'I' )
    ( "NumRich1Hits",      'I' )
    ( "NumRich2Hits",      'I' )
    ( "TrackType",         'I' )
    ( "TrackHistory",      'I' )
    ( "TrackNumDof",       'I' )
    ( "TrackFitVeloNDoF",  'I' )
    ( "TrackFitTNDoF",     'I' )
    ( "RichUsedAero",      'I' )
    ( "RichUsedR1Gas",     'I' )
    ( "RichUsedR2Gas",     'I' )
    ( "RichAboveElThres",  'I' )
    ( "RichAboveMuThres",  'I' )
    ( "RichAbovePiThres",  'I' )
    ( "RichAboveKaThres",  'I' )
    ( "RichAbovePrThres",  'I' )
    ( "InAccMuon",         'I' )
    ( "MuonIsMuon",        'I' )
    ( "MuonIsLooseMuon",   'I' )
    ( "MuonNShared",       'I' )
    ( "InAccEcal",         'I' )
    ( "InAccHcal",         'I' )
    ( "InAccPrs",          'I' )
    ( "InAccSpd",          'I' )
    ( "InAccBrem",         'I' )
    ;
  TMVANodeTypeMap::const_iterator i = map.find(var);
  return ( i == map.end() ? defType : i->second );
}


#endif // NODETYPEMAPPING_H
