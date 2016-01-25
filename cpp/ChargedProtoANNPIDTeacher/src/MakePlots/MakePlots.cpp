
// STL
#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <sstream>

// boost
#include "boost/algorithm/string.hpp"
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include "boost/lexical_cast.hpp"
#include "boost/assign/list_of.hpp"

// ROOT
#include <TSystem.h>
#include <TROOT.h>
#include <TCanvas.h>
#include <TH1F.h>
#include <TGraphErrors.h>
#include <TLegend.h>

// Local
#include "ChargedProtoANNPIDTeacher/rootstyle.h"
#include "ChargedProtoANNPIDTeacher/PrintCanvas.h"

static const std::pair<int,int> canvasDims(1280,1024);

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

int getColour( const bool reset = false )
{
  static const std::vector<int> colours = boost::assign::list_of
    (kBlack)(kRed)(kGreen)(kBlue)(kMagenta)(kCyan)(kOrange)
    (kAzure)(kViolet)(kPink)(kYellow)(kSpring)(kGray)(kTeal)
    ;
  static std::vector<int>::const_iterator i = colours.begin();
  if ( reset || i == colours.end() )
  {
    //std::cerr << "Warning - Run out of colours ...." << std::endl;
    i = colours.begin();
  }
  return *(i++);
}

int getMarker( const bool reset = false )
{
  static const std::vector<int> markers = boost::assign::list_of
    (20)(21)(22)(23)(29)(33)(34)(24)(25)(26)(27)(28)(30)(31)(32)
    ;
  static std::vector<int>::const_iterator i = markers.begin();
  if ( reset || i == markers.end() )
  {
    //std::cerr << "Warning - Run out of markers ...." << std::endl;
    i = markers.begin();
  }
  return *(i++);
}

int main(int argc, char** argv)
{
  if ( argc < 3 || argc > 5 )
  {
    std::cout << "Wrong number of arguments" << std::endl;
    return 1;
  }

  const std::string target_path(argv[1]);
  const std::string type(argv[2]);

  const std::string optFilt = ( argc >= 4 ? argv[3] : "(.*)" );

  const std::vector<std::string> tracks = boost::assign::list_of
    ("Long")("Downstream")("Upstream");
  const std::vector<std::string> particles = boost::assign::list_of
    ("Electron")("Muon")("Pion")("Kaon")("Proton")("Ghost");

  for ( std::vector<std::string>::const_iterator track = tracks.begin();
        track != tracks.end(); ++track )
  {
    for ( std::vector<std::string>::const_iterator particle = particles.begin();
          particle != particles.end(); ++particle )
    {

      const boost::regex my_filter( *track+*particle+"EffV"+type+"-"+optFilt+".txt" );
      std::cout << "Making Plots in " << target_path
                << " for " << my_filter << std::endl;

      typedef std::vector<std::string> Files;
      Files files;

      boost::filesystem::recursive_directory_iterator end_itr;
      for ( boost::filesystem::recursive_directory_iterator i(target_path); 
            i != end_itr; ++i )
      {
        // Skip if not a file
        if ( !boost::filesystem::is_regular_file(i->status()) ) continue;

        // Skip if no match
        boost::cmatch matches;
        if ( !boost::regex_match( i->path().filename().string().c_str(),
                                  matches, my_filter ) ) continue;

        // File matches, store it
        files.push_back( i->path().parent_path().string() + "/" + i->path().filename().string() );
        std::cout << "Matched " << files.back() << std::endl;
      }

      std::sort( files.begin(), files.end() );

      if ( !files.empty() )
      {

        // root style
        setStyle();

        // Make a canvas
        TCanvas * c = new TCanvas( ((*track)+(*particle)+type).c_str(), 
                                   ((*track)+(*particle)+type).c_str(), 
                                   canvasDims.first, canvasDims.second );
        c->SetGrid();

        bool firstPlot = true;

        // Define the legion
        TLegend * l = ( "Purity" == type ?
                        new TLegend( 0.1, 0.1, 0.5, 0.1 + ( 0.014 * files.size() ) ) :
                        new TLegend( 0.1, 0.9, 0.5, 0.9 - ( 0.014 * files.size() ) ) );
        l->SetTextSize(0.012);
        l->SetMargin(0.05);

        // Loop over data files
        for ( Files::const_iterator iF = files.begin(); iF != files.end(); ++iF )
        {

          // Read the data from the text file
          std::cout << "Reading data from " << *iF << std::endl;
          std::ifstream dataS( (*iF).c_str() );
          if ( dataS.is_open() )
          {
            // regex expression for lines with 4 values
            const boost::regex expr("(.*) (.*) (.*) (.*)");

            // First few lines are general data
            std::string title,PartTrackType,xtitle,ytitle,limits;
            std::getline(dataS,title);
            std::getline(dataS,PartTrackType);
            std::getline(dataS,xtitle);
            std::getline(dataS,ytitle);
            std::getline(dataS,limits);
            double xmin(0),xmax(100),ymin(0),ymax(100);
            boost::cmatch lmatches;
            if ( boost::regex_match( limits.c_str(), lmatches, expr ) )
            {
              xmin = boost::lexical_cast<double>(lmatches[1]);
              xmax = boost::lexical_cast<double>(lmatches[2]);
              ymin = boost::lexical_cast<double>(lmatches[3]);
              ymax = boost::lexical_cast<double>(lmatches[4]);
            }
            boost::replace_all(title,"-"," ");

            // Get remaining (x,y) data
            std::vector<double> x,xerr,y,yerr;
            std::string line;
            while( std::getline(dataS,line) )
            {
              boost::cmatch matches;
              if ( boost::regex_match( line.c_str(), matches, expr ) )
              {
                x   .push_back( boost::lexical_cast<double>(matches[1]) );
                xerr.push_back( boost::lexical_cast<double>(matches[2]) );
                y   .push_back( boost::lexical_cast<double>(matches[3]) );
                yerr.push_back( boost::lexical_cast<double>(matches[4]) );
              }
            }

            const int colour = getColour(firstPlot);
            const int marker = getMarker(firstPlot);

            // Draw on the canvas
            TGraphErrors * g = new TGraphErrors( x.size(),
                                                 &*x.begin(),
                                                 &*y.begin(),
                                                 &*xerr.begin(),
                                                 &*yerr.begin() );
            setGraphLimits( g, xmin, xmax, ymin, ymax );
            g->SetTitle( PartTrackType.c_str() );
            g->GetXaxis()->SetTitle( xtitle.c_str());
            g->GetYaxis()->SetTitle( ytitle.c_str());
            g->SetMarkerColor(colour);
            g->SetLineColor(colour);
            g->SetMarkerStyle(marker);

            l->AddEntry( g, title.c_str(), "p" );

            // Draw it
            g->Draw( firstPlot ? "ALP" : "LP" );
            firstPlot = false;

          }
          else
          {
            std::cerr << "Failed to open " << *iF << std::endl;
          }

        }

        // Draw legion last
        l->Draw();

        // Make plots
        std::string name = *track+"_"+*particle+"_"+type;

        c->SetLogy(false);
        printCanvas( c, name+"_Liny" );
        //c->SetLogy(true);
        //printCanvas( c, name+"_Logy" );

        delete l;

      }

    }
  }

  return 0;
}
