
// Local
#include "ChargedProtoANNPIDTeacher/PrintCanvas.h"

// boost
#include <boost/filesystem.hpp>
#include "boost/assign/list_of.hpp"

void printCanvas( TCanvas * c, const std::string & name )
{
  // Image file suffices
  static const std::vector<std::string> imageTypes =
    boost::assign::list_of("png")("pdf");
  if ( c )
  {
    for ( std::vector<std::string>::const_iterator type = imageTypes.begin();
          type != imageTypes.end(); ++type )
    {
      boost::filesystem::path fullName( *type + "/" + name + "." + *type );
      if ( boost::filesystem::exists(fullName.parent_path()) ||
           boost::filesystem::create_directory(fullName.parent_path()) )
      {
        c->Print( fullName.string().c_str(), (*type).c_str() );
      }
    }
  }
}
