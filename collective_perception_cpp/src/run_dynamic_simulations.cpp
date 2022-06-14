#include <argos3/core/simulator/simulator.h>
#include <argos3/core/utility/plugins/dynamic_loading.h>
#include <argos3/core/simulator/query_plugins.h>
#include <argos3/core/simulator/argos_command_line_arg_parser.h>

#include <argos3/core/simulator/loop_functions.h>

using namespace argos;

int main(int n_argc, char **ppch_argv)
{
   try
   {
      /* Create a new instance of the simulator */
      CSimulator &cSimulator = CSimulator::GetInstance();

      /* Configure the command line options */
      CARGoSCommandLineArgParser cACLAP;

      /* Parse command line */
      cACLAP.Parse(n_argc, ppch_argv);

      switch (cACLAP.GetAction())
      {
      case CARGoSCommandLineArgParser::ACTION_RUN_EXPERIMENT:
      {
         CDynamicLoading::LoadAllLibraries();

         cSimulator.SetExperimentFileName(cACLAP.GetExperimentConfigFile());
         cSimulator.LoadExperiment();

         while (!cSimulator.GetLoopFunctions().IsExperimentFinished())
         {
            cSimulator.Reset();
            cSimulator.Execute();
         }
         break;
      }
      case CARGoSCommandLineArgParser::ACTION_UNKNOWN:
      {
         /* Should never get here */
         break;
      }
      }

      /* Done, destroy stuff */
      cSimulator.Destroy();
   }
   catch (std::exception &ex)
   {
      /* A fatal error occurred: dispose of data, print error and exit */
      LOGERR << ex.what() << std::endl;
#ifdef ARGOS_THREADSAFE_LOG
      LOG.Flush();
      LOGERR.Flush();
#endif
      return 1;
   }
   /* Everything's ok, exit */
   return 0;
}