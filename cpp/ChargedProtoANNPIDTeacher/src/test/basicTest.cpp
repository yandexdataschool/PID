
// NeuroBayes
#include "NeuroBayesExpert.hh"

// STL
#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
  
int main(int argc, char** argv)
{
  if ( argc != 3 )
  {
    std::cout << "Wrong number of arguments" << std::endl;
    return 1;
  }
  
  // network config file
  const std::string configFileName(argv[1]);
  std::cout << "Config file " << configFileName << std::endl;
  // data
  const std::string dataFileName(argv[2]);
  std::cout << "Data file   " << dataFileName   << std::endl;

  // Open text data file

  std::ifstream dataFile( dataFileName.c_str() );
  if ( !dataFile.is_open() )
  {
    std::cerr << "Failed to open datafile" << std::endl;
    return 1;
  }

  // First line is number of inputs
  int nInputs(0);
  dataFile >> nInputs;
  std::cout << "Number of inputs = " << nInputs << std::endl;

  // Allocate a 'large' object' to attempt move in memory ...
//   const int nDumpSize = 1000;
//   double * dum = new double[nDumpSize];
//   for ( int i = 0; i < nDumpSize; ++i ) { dum[i] = double(i); }

  // Make an expert
  Expert * expert = new Expert(configFileName.c_str());

  // input array
  float InputArray[nInputs];

  // basic stats
  double totalOutputs(0);
  unsigned int nOutputs(0);

  // Loop over the data inputs
  float var(0);
  int input(0);
  float targetnnOut(0);
  unsigned int nPrintout(0);
  while ( dataFile >> var )
  {
    if ( 0 == input )
    {
      // first value on each line is the NN output from the training evaluation
      targetnnOut = var;
      // std::cout << "Target output = " << var << std::endl;
    }
    else
    {
      // remaining variables are inputs in order
      InputArray[input-1] = var;
      //std::cout << " -> Input " << input-1 << " = " << var << std::endl;
    }
    if ( ++input == nInputs+1 )
    {
      const double nnOut = ( 1.0 + expert->nb_expert(InputArray) ) / 2.0;
      input = 0;
      const float diff = nnOut-targetnnOut;
      //if ( fabs(diff) > 0.0001 )
      if ( ++nPrintout < 100 )
      {
        std::cout << " New Output = " << nnOut 
                  << " Old Output = " << targetnnOut
                  << " Diff = " << diff
                  << std::endl;
      }
      totalOutputs += nnOut;
      ++nOutputs;
    }
  }

  // clean up
  delete expert;
  //delete[] dum;
  dataFile.close();

  return 0;
}
