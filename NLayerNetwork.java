import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.Math;
import java.util.stream.Stream;

/**
 * @author Saanvi Bhargava
 * Created: April 9, 2024
 * 
 * An N Layer neural network that has N hidden layers. 
 * The hidden layers, input layer, and output layer can have any number of activations.
 * Computes an output based on initial input activations, configured in the code itself, along with the weights between layers.
 * The network trains by gradient descent with backpropagation and uses the sigmoid activation function. 
 * The computed output is then compared to the expected output using an error function. 
 * After the network is run on all training sets, the net and mean error is calculated. 
 * Methods include:
 * public static void main(String[] args) throws Exception 
 * public static void setConfigParams() throws IOException
 * public static void echoConfigParams()
 * public static void allocateArrMem()
 * public static void populateArrays() throws Exception
 * public static void populateTruthTable() throws IOException
 * public static void train() throws IOException
 * public static double thresholdFunc(double theta)
 * public static double sigmoidFunc(double theta)
 * public static double hyperbolicTanFunc(double theta)
 * public static double derivThresholdFunc(double F)
 * public static double derivSigmoidFunc(double F)
 * public static void runNetwork() throws IOException
 * public static void populateTestCases() throws IOException
 * public static void runForTrain(int numTestCase)
 * public static void runTruthTableTestCase(int testCaseNum)
 * public static void runTestCase(int testCaseNum)
 * public static int findLargestLayer(String configString)
 * public static void printTruthTable()
 * public static void printExpTruthTable()
 * public static void reportResults()
 * public static double genRandWeight()
 * public static void saveWeights() throws IOException
 * public static void loadWeights() throws Exception
 */
public class NLayerNetwork
{
   static double lambda;                     // the learning factor used to control the magnitude of weight changes
   static boolean training;                  // true if the network is in training mode, false if the network is in running mode
   static int numInputActivations;           // number of input activations
   static int numHiddenLayers;               // number of hidden layers
   static int numLayers;                     // number of total layers in the network
   static int numActivations[];              // number of activations in each layer layer
   static int numOutputActivations;          // number of output activations
   static double lowerrand;                  // minimum randomized weight value
   static double higherrand;                 // maximum randomized weight value
   static int maxIterations;                 // maximum number of iterations allowed before training is stopped
   static double errorThreshold;             // maximum error tolerance for training completion
   static double errorReached;               // minimum error achieved at completion of training
   static int numTestCases;                  // the number of training sets 
   static boolean applyThresholdFunc;        // true if threshold function is to be used for the activation function
   static double weights[][][];              // weights array that stores weights for all layers
   static double activations[][];            // activations array that stores input, hidden, and output activations
   static double truthTableInputs[][];       // truth table with each test case and the input activations
   static double truthTableExpOutputs[][];   // truth table with the expected output activations
   static double testCaseInputs[][];         // test case data with input test cases
   static double actOutputs[][];             // array for actual output activations for all test cases
   static int numInterations;                // number of iterations training took to reach the error threshold
   static double psis[][];                   // psi array to assist with weight changes as defined by the design documents
   static String weightsFile;                // file name for weights
   static double thetas[][];                 // theta array to assist with weight changes as defined by the design document
   static String configFile;                 // file name for configuration file with params to be used at runtime
   static String testCaseFile;               // file name for file with test case inputs and expected outputs
   static String truthTableFile;             // file name for file with truth table for training
   static long runStart;                     // start time for running the network in milliseconds, used to print elapsed time
   static long runFinish;                    // end time for running the network in milliseconds, used to print elapsed time
   static long trainStart;                   // start time for training the network in milliseconds, used to print elapsed time
   static long trainFinish;                  // end time for training the network in milliseconds, used to print elapsed time
   static int loadWeights;                   // loads weights, randomizes, or uses pre-loaded depending on number
   static boolean saveWeights;               // saves weights if true, does not save weights if false
   static String networkConfigString;        // network configuration in the form A-B-C
   static int outputLayer;                   // index for the output layer in the activations array
   static int keepAlive;                     // number of iterations to print after
   
   public static final String DEFAULT_CONFIG_FILE = "paramsN.cfg";   // default configuration file name
   
   public static final int HARDCODE_WEIGHTS = 0;      // tells the network that if loadWeights == 0, it should hardcode weights
   public static final int RANDOMIZE_WEIGHTS = 1;     // tells the network that if loadWeights == 1, it should randomize weights
   public static final int LOAD_WEIGHTS = 2;          // tells the network that if loadWeights == 2, it should load weights
   public static final int INPUT_LAYER = 0;           // constant for index of input layer
   public static final int HIDDEN_LAYER_1 = 1;        // constant for index of first hidden layer
   public static final int HIDDEN_LAYER_2 = 2;        // constant for index of second hidden layer

   public static void main(String[] args) throws Exception 
   {
      if (args.length > 0)
      {
         for (int i = 0; i < args.length; ++i) 
         {
            configFile = args[i].strip();
         }
      }
      else
      {
         configFile = DEFAULT_CONFIG_FILE;
         System.out.println("No config file passed, using the default file " + configFile);
      }
         
      setConfigParams();      // configures network parameters
      echoConfigParams();     // prints network parameters and checks for issues
      allocateArrMem();       // configures space for the arrays to be used for the network
      populateArrays();       // populates weights arrays and truth table
      
      if (training)
      {
         train();             // trains the network to converge to a maximum error or number of iterations
         saveWeights();
         runNetwork();
         
      } // if (training)

      else 
      {
         runNetwork();        // runs the network on the test cases
      }
      reportResults();        // reports expected output from running results

   } // public static void main(String[] args) 

/**
 * Initializes all the instance variables, excluding arrays. Takes values from file specified by user.
 * @throws IOException 
 */
   public static void setConfigParams() throws IOException
   {
      FileReader r = new FileReader(configFile);
      BufferedReader reader = new BufferedReader(r);
      String line;

      while ((line = reader.readLine()) != null) 
      {
         if (line.startsWith("#"))
         {
            continue;
         }

         else
         {
            String varName = line.split("=")[0].strip();
            String value = line.split("=")[1].strip();

            switch (varName) {
               case "lambda":
                  lambda = Double.parseDouble(value);
                  break;
               case "training":
                  training = Boolean.parseBoolean(value);
                  break;
               case "numInputActivations":
                  numInputActivations = Integer.parseInt(value);
                  break;
               case "numOutputActivations":
                  numOutputActivations = Integer.parseInt(value);
                  break;
               case "numLayers":
                  numLayers = Integer.parseInt(value);
                  numHiddenLayers = numLayers - 2;
                  outputLayer = numLayers - 1;
                  break;
               case "networkConfig":
                  networkConfigString = value;
                  break;
               case "lowerrand":
                  lowerrand = Double.parseDouble(value);
                  break;
               case "higherrand":
                  higherrand = Double.parseDouble(value);
                  break;
               case "maxIterations":
                  maxIterations = Integer.parseInt(value);
                  break;
               case "errorThreshold":
                  errorThreshold = Double.parseDouble(value);
                  break;
               case "numTestCases":
                  numTestCases = Integer.parseInt(value);
                  break;
               case "applyThresholdFunc":
                  applyThresholdFunc = Boolean.parseBoolean(value);
                  break;
               case "keepAlive":
                  keepAlive = Integer.parseInt(value);
                  break;
               case "weightsFile":
                  weightsFile = value;
                  break;
               case "testCaseFile":
                  testCaseFile = value;
                  break;
               case "truthTableFile":
                  truthTableFile = value;
                  break;
               case "loadWeights":
                  loadWeights = Integer.parseInt(value);
                  break;
               case "saveWeights":
                  saveWeights = Boolean.parseBoolean(value);
                  break;
               default:
                  break;
            } // switch (varName)
         } // else
      } // while ((line = reader.readLine()) != null)

      reader.close();
   } // public static void setConfigParams()

/**
 * Displays the key details of the network configuration such as
 * learning factor, dimensions, random ranges, etc
 */
   public static void echoConfigParams()
   {
      System.out.println("\n\nNetwork Configuration:\n");
      System.out.println("The configuration is a " + networkConfigString + " network.");

      if (training)
      {
         System.out.println("The network is in training mode.");

         if (loadWeights == RANDOMIZE_WEIGHTS)
         {
            System.out.println("The network is starting with random weights between " + lowerrand + " and " + higherrand + ".");
         }
         else if (loadWeights == HARDCODE_WEIGHTS)
         {
            System.out.println("The network is starting training with predetermined weights.");
         }
         else if (loadWeights == LOAD_WEIGHTS)
         {
            System.out.println("The network is loading weights from " + weightsFile + ".");
         }

         if (saveWeights)
         {
            System.out.println("The network is saving weights to " + weightsFile + ".");
         }

         System.out.println("The learning factor (lambda) for gradient descent is " + lambda + ".");
         System.out.print("The training will terminate either once the error is below the threshold of " + errorThreshold);
         System.out.println(" or once the maximum number of training iterations, " + maxIterations + ", is completed.");
      } // if (training)
      else
      {
         System.out.println("The network is in running mode.");
      }
      
      System.out.print("The network has " + numInputActivations + " input activations, ");
      System.out.println((numHiddenLayers) + " hidden layers, and " +  numOutputActivations + " output activation.\n\n");
   } // public static void echoConfigParams()

/**
 * Initializes arrays with appropriate dimensions. Does not populate them.
 */
   public static void allocateArrMem()
   {
      int dim = findLargestLayer(networkConfigString);
      numActivations = new int[numLayers];

      activations = new double[numLayers][dim];

      truthTableInputs = new double[numTestCases][numInputActivations]; 
      truthTableExpOutputs = new double[numTestCases][numOutputActivations]; 
      actOutputs = new double[numTestCases][numOutputActivations]; 
      testCaseInputs = new double[numTestCases][numInputActivations];
      weights = new double[numLayers][dim][dim];
      
      if (training)
      {
         psis = new double[numLayers][dim];
         thetas = new double[numLayers][dim];
      } // if (training)
      
   } // public static void allocateArrMem()

/**
 * Populating the weight arrays and truth table with initial values.
 * Weight array values depends on randomization and truth table is currently manually entered.
 * @throws Exception 
 */
   public static void populateArrays() throws Exception
   {  
      numActivations = Stream.of(networkConfigString.split("-")).mapToInt(Integer::parseInt).toArray();

/**
 * populating weights
 */ 
      if (loadWeights == LOAD_WEIGHTS)
      {
         loadWeights();
      }
      else if (loadWeights == RANDOMIZE_WEIGHTS)
      {
         for (int n = 0; n < outputLayer; n++)
         {
            for (int b = 0; b < numActivations[n]; b++)
            {
               for (int g = 0; g < numActivations[n+1]; g++)
               {
                  weights[n][b][g] = genRandWeight();
               }
            }
         } // for (int n = 0; n < outputLayer; n++)
      } // if (randomize)
      else if (loadWeights == HARDCODE_WEIGHTS)
      {
/**
 * filling pre-loaded weights
 */
         weights[0][0][0] = 0.8;
         weights[0][1][0] = 0.6;
         weights[0][0][1] = 0.2;
         weights[0][1][1] = 0.9;
         weights[0][0][0] = 0.4;
         weights[0][1][0] = 0.1;
      }
      
      populateTruthTable();
      populateTestCases();
   } // public static void populateArrays()

/**
 * Populates test case inputs and expected outputs into truth table arrays from file. 
 * @throws IOException 
 */
   public static void populateTruthTable() throws IOException
   {
      FileReader r = new FileReader(truthTableFile);
      BufferedReader reader = new BufferedReader(r);
      String line;
      int iter = 0;
      String[] testCaseData;

      while ((line = reader.readLine()) != null) 
      {
         if (line.startsWith("#"))
         {
            continue;
         }
         
         testCaseData = line.split(" ");

         for (int inAct = 0; inAct < numInputActivations; inAct++)
         {
            truthTableInputs[iter][inAct] = Double.parseDouble(line.split(" ")[inAct]);
         }

         for (int outAct = 0; outAct < numOutputActivations; outAct++)
         {
            truthTableExpOutputs[iter][outAct] = Double.parseDouble(testCaseData[numInputActivations+outAct]);
         }

         iter++;
      } // while ((line = reader.readLine()) != null) 

      reader.close();
   } // public static void populateTruthTable() throws IOException

/**
 * This method is responsible for training the perceptron by updating its weights to minimize the error.
 * Calculates the value of the activations in the hidden layers using dot products of the weights of the input layers.
 * It employs the perceptron learning algorithm, 
 * iterating through the test cases until convergence or reaching a maximum number of iterations.
 * The convergence is determined by the error falling below a specified threshold.
 * @throws IOException 
 */
   public static void train() throws IOException
   {
      trainStart = System.currentTimeMillis();
      double avgErr = 0.0;
      int iter = 0;
      boolean done = false;
      double E;
      double diff = 0.0; 
      double omegaA;
      int a, b, g;

      while (!done) // looping until end condition is met (error is below threshold or max iterations is reached)
      {
         double totalErr = 0.0;

         for (int testCase = 0; testCase < numTestCases; testCase++) // in each iteration, looping through all training cases
         {

            for (int k = 0; k < numInputActivations; k++)
            {
               activations[INPUT_LAYER][k] = truthTableInputs[testCase][k];
            }

            runForTrain(testCase);

            for (a = outputLayer; a > HIDDEN_LAYER_2; a--)
            {
               for (b = 0; b < numActivations[a-1]; b++)
               {
                  omegaA = 0.0;

                  for (g = 0; g < numActivations[a]; g++)
                  {
                     omegaA += psis[a][g] * weights[a-1][b][g];
                     weights[a-1][b][g] += lambda * activations[a-1][b] * psis[a][g];
                  }
                  
                  psis[a-1][b] = omegaA * derivThresholdFunc(thetas[a-1][b]);
               } // for (b = 0; b < numActivations[a-1]; b++)
            } // for (a = outputLayer; a >= HIDDEN_LAYER_2; a--)

            a = HIDDEN_LAYER_2;

            for (b = 0; b < numActivations[a-1]; b++)
            {
               omegaA = 0.0;
               
               for (g = 0; g < numActivations[a]; g++)
               {
                  omegaA += psis[a][g] * weights[a-1][b][g];
                  weights[a-1][b][g] += lambda * activations[a-1][b] * psis[a][g];
               }

               psis[a-1][b] = omegaA * derivThresholdFunc(thetas[a-1][b]);

               for (g = 0; g < numActivations[a-2]; g++)
               {
                  weights[a-2][g][b] += lambda * activations[a-2][g] * psis[a-1][b];
               }
            } // for (b = 0; b < numActivations[a-1]; b++)
            
            runTruthTableTestCase(testCase); // running network again to determine error with new weights
            
            E = 0.0; 
            for (int outAct = 0; outAct < numOutputActivations; outAct++)
            {
               diff = (truthTableExpOutputs[testCase][outAct] - activations[outputLayer][outAct]);
               E += diff*diff;
            }
            E *= 0.5;
            totalErr += E; // adding to total error for all test cases in this iteration
             
         } // for (int testCase = 0; testCase < numTestCases; testCase++)

         iter++;
         avgErr = (totalErr/numTestCases);
         done = (avgErr < errorThreshold || iter >= maxIterations); // defining done
      
         if ((keepAlive > 0) && (iter % keepAlive == 0)) 
         {
            System.out.printf("Iteration %d, Error = %f\n", iter, avgErr);
         }
            
      } // while (!done)
      
      errorReached = avgErr; // final error achieved below minimum error threshold or because of completion of iterations
      numInterations = iter;
      trainFinish = System.currentTimeMillis();
      return;
   } // public static void train()

/**
 * If the threshold wrapper function is to be applied, this method applies the function to activation for weight calculation.
 */
public static double thresholdFunc(double theta)
{
   if (applyThresholdFunc)
   {
      return sigmoidFunc(theta);
   }
   return theta;
} // public static double thresholdFunc(double theta)

/**
* Sigmoid function for the threshold wrapper function. 
* The sigmoid function is as follows: f(x) = 1/(1+e^(-x))
*/
public static double sigmoidFunc(double theta)
{
   double efunc = Math.exp((-theta));
   theta = 1.0/(1.0 + efunc);
   return theta;
} // public static double sigmoidFunc(double theta)

/**
 * Finds the hyperbolic tan value for a given theta for the threshold wrapper function.
 * @param theta The value to find hyperbolic tan of.
 * @return Hyperbolic tan of theta
 */
public static double hyperbolicTanFunc(double theta)
{
   double efunc = Math.exp(theta);
   double negefunc = 1.0/efunc;

   theta = (efunc - negefunc)/(efunc + negefunc);

   return theta;
} // public static double hyperbolicTanFunc(double theta)

/**
* Calculates the value of the derivative of the threshold wrapper function for F. 
*/
public static double derivThresholdFunc(double F)
{
   return derivSigmoidFunc(F);
} // public static double derivThresholdFunc(double F)

/**
* Calculates the value of the derivative of the sigmoid wrapper function for F. 
*/
public static double derivSigmoidFunc(double F)
{
   double x = thresholdFunc(F);
   return x * (1.0 - x);
} // public static double derivThresholdFunc(double F)

/**
 * Runs the network on all test cases in the truth table and updates the truth table with the determined output.
 * @throws IOException 
 */
   public static void runNetwork() throws IOException
   {
      runStart = System.currentTimeMillis();

      System.out.println("\nBeginning predictions.");

      for (int testCase = 0; testCase < numTestCases; testCase++)
      {
         runTestCase(testCase);
      }

      runFinish = System.currentTimeMillis();
   } // public static void runNetwork()

   /**
    * Populates test cases for running the network.
    * @throws IOException
    */
   public static void populateTestCases() throws IOException
   {
      FileReader r = new FileReader(testCaseFile);
      BufferedReader reader = new BufferedReader(r);
      String line;
      int iter = 0;
      String[] testCaseData;

      while ((line = reader.readLine()) != null) 
      {
         if (line.startsWith("#"))
         {
            continue;
         }
         
         testCaseData = line.split(" ");

         for (int inAct = 0; inAct < numInputActivations; inAct++)
         {
            testCaseInputs[iter][inAct] = Double.parseDouble(testCaseData[inAct]);
         }

         iter++;
      } // while ((line = reader.readLine()) != null) 

      reader.close();
   } // public static void populateTestCases() throws IOException


/**
 * Runs test case given in parameter for the training of the network. Determines psii.
 */
   public static void runForTrain(int numTestCase)
   {
      double thetaAB;
      int a, b, g;

      for (a = 1; a < outputLayer; a++)
      { 
         for (b = 0; b < numActivations[a]; b++)
         {
            thetas[a][b] = 0.0;
            
            for (g = 0; g < numActivations[a-1]; g++)
            {
               thetas[a][b] += activations[a-1][g] * weights[a-1][g][b];
            }

            activations[a][b] = thresholdFunc(thetas[a][b]);
         } // for (b = 0; b < numActivations[a]; b++)
      } // for (a = 1; a < outputLayer; a++)
      
      a = outputLayer;

      for (b = 0; b < numActivations[a]; b++)
      {
         thetaAB = 0.0;
         
         for (g = 0; g < numActivations[a-1]; g++)
         {
            thetaAB += activations[a-1][g] * weights[a-1][g][b];
         }

         activations[a][b] = thresholdFunc(thetaAB);
         psis[a][b] = (truthTableExpOutputs[numTestCase][b] - activations[a][b]) * derivThresholdFunc(thetaAB);
      } // for (b = 0; b < numActivations[a]; b++)

   } // public static void runForTrain(int numTestCase)

/**
 * Runs the network on a singular test case. Calculates weights and then F0.
 */
   public static void runTruthTableTestCase(int testCaseNum)
   {
      int a, b, g;
      double thetaAB;

/**
 * calculates hidden activations based on weights and input activation
 */
      for (a = 1; a < outputLayer; a++)
      { 
         for (b = 0; b < numActivations[a]; b++)
         {
            thetaAB = 0.0;
            
            for (g = 0; g < numActivations[a-1]; g++)
            {
               thetaAB += activations[a-1][g] * weights[a-1][g][b];
            }

            activations[a][b] = thresholdFunc(thetaAB);
         } // for (b = 0; b < numActivations[a]; b++)
      } // for (a = 1; a < outputLayer; a++) 

      a = outputLayer;
/**
 * determining F0 based on hidden activations and weights
 */
      for (b = 0; b < numActivations[a]; b++)
      {
         thetaAB = 0.0;
         for (g = 0; g < numActivations[a-1]; g++)
         {
            thetaAB += weights[a-1][g][b] * activations[a-1][g];
         }

         actOutputs[testCaseNum][b] = thresholdFunc(thetaAB);
      } // for (b = 0; b < numActivations[a]; b++)
      
   } // public static double runTruthTableTestCase(int testCaseNum)

/**
 * Run a singular test case
 * @param testCaseNum The test case number to run
 */
   public static void runTestCase(int testCaseNum)
   {
      int a, b, g;
      double thetaAB;

      for (int m = 0; m < numActivations[INPUT_LAYER]; m++)
      {
         activations[INPUT_LAYER][m] = testCaseInputs[testCaseNum][m];
      }

/**
 * calculates hidden activations based on weights and input activation
 */
      for (a = 1; a < outputLayer; a++)
      { 
         for (b = 0; b < numActivations[a]; b++)
         {
            thetaAB = 0.0;
            for (g = 0; g < numActivations[a-1]; g++)
            {
               thetaAB += activations[a-1][g] * weights[a-1][g][b];
            }

            activations[a][b] = thresholdFunc(thetaAB);
         } // for (b = 0; b < numActivations[a]; b++)
      } // for (a = 1; a < outputLayer; a++)

/**
 * determining F0 based on hidden activations and weights
 */
      for (b = 0; b < numActivations[a]; b++)
      {
         thetaAB = 0.0;
         for (g = 0; g < numActivations[a-1]; g++)
         {
            thetaAB += weights[a-1][g][b] * activations[a-1][g];
         }

         actOutputs[testCaseNum][b] = thresholdFunc(thetaAB);
      } // for (b = 0; b < numActivations[a]; b++)
      
   } // public static double runTestCase(int testCaseNum)

/**
 * Finds maximum number of activations in any layer from the config string
 * @param configString The network configuration string
 * @return the maximum number of activations in any layer in the network
 */
   public static int findLargestLayer(String configString)
   {
      int m = 0;
      String[] nums = configString.split("-");

      for (int x = 0; x < nums.length; x++) 
      {
         m = Math.max(m, Integer.parseInt(nums[x]));
      }

      return m;
   } // public static int findLargestLayer(String configString)

/**
 * Prints the entire truth table.
 */
   public static void printTruthTable()
   {
      System.out.println("\nTruth table:\n");

      String headerStr = "Test Case #\tInputs";

      for (int numInputs = 0; numInputs < numInputActivations; numInputs++)
      {
         headerStr += "\t";
      }

      headerStr += "Actual Outputs";

      System.out.println(headerStr);

      for (int t = 0; t < numTestCases; t++)
      {
         String printStr = " " + (t+1) + ": \t\t";

         for (int numIn = 0; numIn < numInputActivations; numIn++)
         {
            printStr += testCaseInputs[t][numIn] + "\t";
         }

         for (int numOut = 0; numOut < numOutputActivations; numOut++)
         {
            printStr += actOutputs[t][numOut] + "\t";
         }

         System.out.println(printStr);
      } // for (int t = 0; t < numTestCases; t++)
   } // public static void printTruthTable()

/**
 * Prints expected truth table.
 */
   public static void printExpTruthTable()
   {
      System.out.println("\nTruth table:\n");

      String headerStr = "Test Case #\tInputs";

      for (int numInputs = 0; numInputs < numInputActivations; numInputs++)
      {
         headerStr += "\t";
      }

      headerStr += "Expected Output";
      System.out.println(headerStr);

      for (int t = 0; t < numTestCases; t++)
      {
         String printStr = " " + (t+1) + ": \t\t";

         for (int numIn = 0; numIn < numInputActivations; numIn++)
         {
            printStr += truthTableInputs[t][numIn] + "\t";
         }

         for (int numOut = 0; numOut < numOutputActivations; numOut++)
         {
            printStr += truthTableExpOutputs[t][numOut] + "\t";
         }

         System.out.println(printStr);
      } // for (int t = 0; t < numTestCases; t++)
   } // public static void printTruthTable()

/**
 * Reports the results based on training and running. 
 * For training, the results include weights, reason for termination, and error achieved.
 * For running, the results include the truth table containing predicted outputs.
 */
   public static void reportResults()
   {
      System.out.println("Threshold function being used is sigmoid function.");

      if (training)
      {
         System.out.println("Training is completed. Training results are:");
         System.out.println("\nThe final error was " + errorReached + ".");
         System.out.println("Training terminated in " + (trainFinish - trainStart) + " ms.");

         if (errorReached < errorThreshold)
         {
            System.out.println("Training terminated since error threshold was reached after " + numInterations + " iterations.");
         }
         else 
         {
            System.out.println("Training terminated because it completed max iterations.");
         }
         printExpTruthTable();
      } // if (training)

      System.out.println("Running terminated in around " + (runFinish - runStart) + " ms.");
      System.out.println("\nFinished running the network. Predictions are:\n");
      printTruthTable();
   } // public static void reportResults()

/**
 * Generates random weight between a lower and upper bound (inclusive of the lower bound but exclusive of the upper bound).
 */
   public static double genRandWeight()
   {
      return ((higherrand - lowerrand) * Math.random()) + lowerrand;
   } // public static double genRandWeight()

/**
 * Saves weight arrays in text file.
 * @throws IOException 
 */
   public static void saveWeights() throws IOException
   {
      FileWriter w = new FileWriter(weightsFile);

      w.write("# Network Config Info\n");
      w.write(networkConfigString + "\n");
      w.write("# Weights" + "\n");

      for (int n = 0; n < outputLayer; n++)
      {
         for (int b = 0; b < numActivations[n]; b++)
         {
            for (int g = 0; g < numActivations[n+1]; g++)
            {
               w.write(String.valueOf(weights[n][b][g]) + "\n");
            }
         } // for (int b = 0; b < numActivations[n]; b++)
      } // for (int n = 0; n < numLayers-1; n++)

      w.close();
   } // public static void saveWeights() throws IOException

/**
 * Loads weight arrays from text file.
 * @throws Exception 
 */
   public static void loadWeights() throws Exception
   {
      FileReader r = new FileReader(weightsFile);
      BufferedReader reader = new BufferedReader(r);
      String line;
      
      while (!(line = reader.readLine()).startsWith("# Weights"))
      {
         if (line.startsWith("#"))
         {
            continue;
         }
         
         if (!line.strip().equals(networkConfigString))
         {
            throw new Exception("Config file mismatch with weights file.");
         }
      } // while (!(line = reader.readLine()).startsWith("# Weights"))

      for (int n = 0; n < numLayers-1; n++)
      {
         for (int b = 0; b < numActivations[n]; b++)
         {
            for (int g = 0; g < numActivations[n+1]; g++)
            {
               weights[n][b][g] = Double.parseDouble(reader.readLine());
            }
         } // for (int b = 0; b < numActivations[n]; b++)
      } // for (int n = 0; n < numLayers-1; n++)
      reader.close();

   } // public static void loadWeights() throws NumberFormatException, IOException

} // public class Perceptron
