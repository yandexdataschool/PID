#ifndef B2D0HD2KSPIPIDALITZ_NTUPLEREADER_H 
#define B2D0HD2KSPIPIDALITZ_NTUPLEREADER_H 1

// Include files
#include <string>
#include <map>
#include <vector>
#include <iostream>

// ROOT
#include "TTree.h"
#include "TChain.h"
#include "TBranch.h"
#include "TTreeFormula.h"

/** @class NTupleReader NTupleReader.h B2D0hD2KsPiPiDalitz/NTupleReader.h
 *  
 *
 *  @author Chris Jones
 *  @date   2013-01-11
 */
class NTupleReader 
{
public: 

  /// TTree constructor
  NTupleReader( TTree * tree,
                const std::vector<std::string>& inputs );

  /// TChain constructor
  NTupleReader( TChain * chain,
                const std::vector<std::string>& inputs );

  /// Destructor
  ~NTupleReader( ); 

public:

  double variable( const std::string& name ) const;

  inline bool isOK() const { return m_OK; }

  template <class TYPE>
  void addVariable( TYPE * tree,
                    const std::string& name );

private:

  template <class TYPE>
  void initialise( TYPE * tree );

private:

  std::vector<std::string> m_inputs;

  std::map<std::string,TTreeFormula*> m_formulas;

  bool m_OK;

  unsigned int m_index;

};

#endif // B2D0HD2KSPIPIDALITZ_NTUPLEREADER_H
