// Include files 

// STL
#include <sstream>

// local
#include "NTupleReader.h"

// STL
#include <algorithm>
#include <cmath>
#include <exception>

//-----------------------------------------------------------------------------
// Implementation file for class : NTupleReader
//
// 2013-01-11 : Chris Jones
//-----------------------------------------------------------------------------

template <class TYPE>
void NTupleReader::addVariable ( TYPE * tree, const std::string& name )
{
  if ( m_formulas.find(name) == m_formulas.end() )
  {
    std::ostringstream var;
    var << "var" << m_index++;
    m_formulas[name] = new TTreeFormula( var.str().c_str(),
                                         name.c_str(),
                                         tree );
  }
}

template <class TYPE>
void NTupleReader::initialise( TYPE * tree )
{
  // Loop over variables and create a formula object for each one
  for ( std::vector<std::string>::const_iterator iIn = m_inputs.begin();
        iIn != m_inputs.end(); ++iIn )
  {
    addVariable( tree, *iIn );
  }
}

NTupleReader::NTupleReader( TTree * tree,
                            const std::vector<std::string>& inputs )
  : m_inputs(inputs), m_OK(true), m_index(0)
{
  initialise(tree);
}

NTupleReader::NTupleReader( TChain * chain,
                            const std::vector<std::string>& inputs )
  : m_inputs(inputs), m_OK(true), m_index(0)
{
  initialise(chain);
}

// Get a variable
double NTupleReader::variable( const std::string& name ) const
{
  std::map<std::string,TTreeFormula*>::const_iterator i = m_formulas.find(name);
  if ( i == m_formulas.end() )
  {
    std::cerr << "WARNING : Could not compute variable " << name << std::endl;
  }
  return ( i != m_formulas.end() ? i->second->EvalInstance() : -1001 );
}

// Destructor
NTupleReader::~NTupleReader()
{
  for ( std::map<std::string,TTreeFormula*>::iterator i = m_formulas.begin();
        i != m_formulas.end(); ++i ) { delete i->second; }
}

//=============================================================================
