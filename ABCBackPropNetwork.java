import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.Math;
/**
 * @author Saanvi Bhargava
 * Created: March 6, 2024
 * 
 * An A-B-C Back Propagation neural network that has one hidden layer. 
 * The hidden layer, input layer, and output layer can have any number of activations.
 * Computes an output based on initial input activations, configured in the code itself, along with the weights between layers.
 * The network trains by gradient descent with backpropagation and uses the sigmoid activation function. 
 * The computed output is then compared to the expected output using an error function. 
 * After the network is run on all training sets, the net and mean error is calculated. 
 */
public class ABCBackPropNetwork
{
   static double lambda;                     // the learning factor used to control the magnitude of weight changes
   static boolean training;                  // true if the network is in training mode, false if the network is in running mode
   static int numInputActivations;           // number of input activations
   static int numHiddenActivations;          // number of hidden activations in the singular hidden layer
   static int numOutputActivations;          // number of output activations
   static double lowerrand;                  // minimum randomized weight value
   static double higherrand;                 // maximum randomized weight value
   static int maxIterations;                 // maximum number of iterations allowed before training is stopped
   static double errorThreshold;             // maximum error tolerance for training completion
   static double errorReached;               // minimum error achieved at completion of training
   static int numTestCases;                  // the number of training sets 
   static boolean applyThresholdFunc;        // true if threshold function is to be used for the activation function
   static double inputActivations[];         // the input activations in order as defined by the low-level design document
   static double hiddenActivations[];        // the hidden activations in order as defined by the low-level design document
   static double inputToHiddenWeights[][];   // weights array for the calculations from input to hidden activations
   static double hiddenToOutputWeights[][];  // weights array for the calculations from hidden to output activations 
   static double truthTableInputs[][];       // truth table with each test case and the input activations
   static double truthTableExpOutputs[][];   // truth table with the expected output activations
   static double truthTableActOutputs[][];   // truth table with the actual output activations
   static int numInterations;                // number of iterations training took to reach the error threshold
   static double psii[];                     // psii array to assist with weight changes as defined by the design documents
   static double psij[];                     // psij array to assist with weight changes as defined by the design document
   static String weightsFile;                // file name for weights
   static double thetaj[];                   // thetaj array to assist with weight changes as defined by the design document
   static double thetai[];                   // thetai array to assist with weight changes as defined by the design document
   static String configFile;                 // file name for configuration file with params to be used at runtime
   static String testCaseFile;               // file name for file with test case inputs and expected outputs
   static long runStart;                     // start time for running the network in milliseconds, used to print elapsed time
   static long runFinish;                    // end time for running the network in milliseconds, used to print elapsed time
   static long trainStart;                   // start time for training the network in milliseconds, used to print elapsed time
   static long trainFinish;                  // end time for training the network in milliseconds, used to print elapsed time
   static int loadWeights;                   // loads weights, randomizes, or uses pre-loaded depending on number
   static boolean saveWeights;               // saves weights if true, does not save weights if false
   static String networkConfigString;        // network configuration in the form A-B-C

   public static final String DEFAULT_CONFIG_FILE = "defaultparams.cfg";   // default configuration file name
   
   public static final int HARDCODE_WEIGHTS = 0;      // tells the network that if loadWeights == 0, it should hardcode weights
   public static final int RANDOMIZE_WEIGHTS = 1;     // tells the network that if loadWeights == 1, it should randomize weights
   public static final int LOAD_WEIGHTS = 2;          // tells the network that if loadWeights == 2, it should load weights

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
      } // if (training)

      runNetwork();           // runs the network on the test cases
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
               case "numHiddenActivations":
                  numHiddenActivations = Integer.parseInt(value);
                  break; 
               case "numOutputActivations":
                  numOutputActivations = Integer.parseInt(value);
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
               case "weightsFile":
                  weightsFile = value;
                  break;
               case "testCaseFile":
                  testCaseFile = value;
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

      networkConfigString = String.valueOf(numInputActivations) + "-";
      networkConfigString += String.valueOf(numHiddenActivations) + "-" + String.valueOf(numOutputActivations);

      reader.close();
   } // public static void setConfigParams()

/**
 * Displays the key details of the network configuration such as
 * learning factor, dimensions, random ranges, etc
 */
   public static void echoConfigParams()
   {
      System.out.println("\n\nNetwork Configuration:\n");
      System.out.println("The configuration is a " + numInputActivations + "-" + numHiddenActivations + "-" + numOutputActivations + " network.");

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
      System.out.println(numHiddenActivations + " hidden activations, and " +  numOutputActivations + " output activation.\n\n");
   } // public static void echoConfigParams()

/**
 * Initializes arrays with appropriate dimensions. Does not populate them.
 */
   public static void allocateArrMem()
   {
      inputActivations = new double[numInputActivations];
      hiddenActivations = new double[numHiddenActivations];
      truthTableInputs = new double[numTestCases][numInputActivations]; 
      truthTableExpOutputs = new double[numTestCases][numOutputActivations]; 
      truthTableActOutputs = new double[numTestCases][numOutputActivations]; 
      inputToHiddenWeights = new double[numInputActivations][numHiddenActivations];
      hiddenToOutputWeights = new double[numHiddenActivations][numOutputActivations];

      if (training)
      {
         psii = new double[numOutputActivations];
         psij = new double[numHiddenActivations];
         thetai = new double[numOutputActivations];
         thetaj = new double[numHiddenActivations];
      } // if (training)
      
   } // public static void allocateArrMem()

/**
 * Populating the weight arrays and truth table with initial values.
 * Weight array values depends on randomization and truth table is currently manually entered.
 * @throws Exception 
 */
   public static void populateArrays() throws Exception
   {  
/**
 * populating weights
 */ 
      if (loadWeights == LOAD_WEIGHTS)
      {
         loadWeights();
      }
      else if (loadWeights == RANDOMIZE_WEIGHTS)
      {
         
         for (int inAct = 0; inAct < numInputActivations; inAct++)
         {
            for (int hidAct = 0; hidAct < numHiddenActivations; hidAct++)
            {
               inputToHiddenWeights[inAct][hidAct] = genRandWeight();   // rows are inputAct indices; columns are hiddenAct indices
            }
         }
         
         for (int hidAct = 0; hidAct < numHiddenActivations; hidAct++)
         {
            for (int outAct = 0; outAct < numOutputActivations; outAct++)
            {
               hiddenToOutputWeights[hidAct][outAct] = genRandWeight();       // weights for hidden to output calculations
            }
         }

      } // if (randomize)
      else if (loadWeights == HARDCODE_WEIGHTS)
      {
/**
 * filling pre-loaded weights
 */
         inputToHiddenWeights[0][0] = 0.8;
         inputToHiddenWeights[1][0] = 0.6;
         inputToHiddenWeights[0][1] = 0.2;
         inputToHiddenWeights[1][1] = 0.9;
         hiddenToOutputWeights[0][0] = 0.4;
         hiddenToOutputWeights[1][0] = 0.1;
      }
      
      populateTestCases();

   } // public static void populateArrays()

/**
 * Populates test case inputs and expected outputs into truth table arrays from file. 
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
            truthTableInputs[iter][inAct] = Double.parseDouble(line.split(" ")[inAct]);
         }

         for (int outAct = 0; outAct < numOutputActivations; outAct++)
         {
            truthTableExpOutputs[iter][outAct] = Double.parseDouble(testCaseData[numInputActivations+outAct]);
         }

         iter++;
      }

      reader.close();
   }

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
      double omegaj;     

      while (!done) // looping until end condition is met (error is below threshold or max iterations is reached)
      {
         double totalErr = 0.0;

         for (int testCase = 0; testCase < numTestCases; testCase++) // in each iteration, looping through all training cases
         {

            for (int k = 0; k < numInputActivations; k++)
            {
               inputActivations[k] = truthTableInputs[testCase][k];
            }

            runForTrain(testCase);
            
            for (int j = 0; j < numHiddenActivations; j++)
            {
               omegaj = 0.0;

               for (int i = 0; i < numOutputActivations; i++)
               {
                  omegaj += psii[i] * hiddenToOutputWeights[j][i];
                  hiddenToOutputWeights[j][i] += lambda * hiddenActivations[j] * psii[i];
               }

               psij[j] = omegaj * derivThresholdFunc(thetaj[j]);

               for (int k = 0; k < numInputActivations; k++)
               {
                  inputToHiddenWeights[k][j] += lambda * inputActivations[k] * psij[j];
               }
            }

            runTestCase(testCase); // running network again to determine error with new weights
            
            E = 0.0; 
            for (int outAct = 0; outAct < numOutputActivations; outAct++)
            {
               diff = (truthTableExpOutputs[testCase][outAct]-truthTableActOutputs[testCase][outAct]);
               E += diff*diff;
            }
            E *= 0.5;
            totalErr += E; // adding to total error for all test cases in this iteration
            
         } // for (int testCase = 0; testCase < numTestCases; testCase++)

         // if (iter % 1000 == 0)   \debug
         // {
         //    saveWeights();       \debug
         // }

         iter++;
         avgErr = (totalErr/numTestCases);
         done = (avgErr < errorThreshold || iter >= maxIterations); // defining done
      
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
 * Calculates the value of the derivative of the threshold wrapper function for F. 
 */
   public static double derivThresholdFunc(double F)
   {
      return derivSigmoidFunc(F);
   }

/**
 * Calculates the value of the derivative of the sigmoid wrapper function for F. 
 */
   public static double derivSigmoidFunc(double F)
   {
      double x = thresholdFunc(F);
      return x * (1.0 - x);
   }

/**
 * Runs the network on all test cases in the truth table and updates the truth table with the determined output.
 */
   public static void runNetwork()
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
 * Runs test case given in parameter for the training of the network. Determines psii.
 */
   public static void runForTrain(int numTestCase)
   {
      
      for (int j = 0; j < numHiddenActivations; j++)
      {
         thetaj[j] = 0.0;
         for (int k = 0; k < numInputActivations; k++)
         {
            thetaj[j] += inputActivations[k] * inputToHiddenWeights[k][j];
         }

         hiddenActivations[j] = thresholdFunc(thetaj[j]);
      }

      for (int i = 0; i < numOutputActivations; i++)
      {
         thetai[i] = 0.0;
         for (int j = 0; j < numHiddenActivations; j++)
         {
            thetai[i] += hiddenActivations[j] * hiddenToOutputWeights[j][i];
         }

         truthTableActOutputs[numTestCase][i] = thresholdFunc(thetai[i]);
         psii[i] = (truthTableExpOutputs[numTestCase][i] - truthTableActOutputs[numTestCase][i]) * derivThresholdFunc(thetai[i]);
      }

   } // public static void runForTrain(int numTestCase)

/**
 * Runs the network on a singular test case. Calculates weights and then F0.
 */
   public static void runTestCase(int testCaseNum)
   {
      double thetaj;
      double thetai;

/**
 * calculates hidden activations based on weights and input activation
 */
      for (int hidAct = 0; hidAct < numHiddenActivations; hidAct++)
      {
         thetaj = 0.0;
         for (int inAct = 0; inAct < numInputActivations; inAct++)
         {
            thetaj += inputToHiddenWeights[inAct][hidAct] * inputActivations[inAct]; // theta = weight*inputAct for each input act
         }

         hiddenActivations[hidAct] = thresholdFunc(thetaj);
      } // for (int hidAct = 0; hidAct < numHiddenActivations; hidAct++)
      
/**
 * determining F0 based on hidden activations and weights
 */
      for (int outAct = 0; outAct < numOutputActivations; outAct++)
      {
         thetai = 0.0;
         for (int hidAct = 0; hidAct < numHiddenActivations; hidAct++)
         {
            thetai += hiddenToOutputWeights[hidAct][outAct] * hiddenActivations[hidAct];
         }

         truthTableActOutputs[testCaseNum][outAct] = thresholdFunc(thetai);
      }
      
   } // public static double runTestCase(int testCaseNum)

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
            printStr += truthTableInputs[t][numIn] + "\t";
         }

         for (int numOut = 0; numOut < numOutputActivations; numOut++)
         {
            printStr += truthTableActOutputs[t][numOut] + "\t";
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
      } // if (training)
      System.out.println("Running terminated in around " + (runFinish - runStart) + " ms.");
      System.out.println("\nFinished running the network. Predictions are:\n");
      printExpTruthTable();
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

      for (int inAct = 0; inAct < numInputActivations; inAct++)
      {
         for (int hidAct = 0; hidAct < numHiddenActivations; hidAct++)
         {
            w.write(String.valueOf(inputToHiddenWeights[inAct][hidAct]) + "\n");
         }
      }

      for (int hidAct = 0; hidAct < numHiddenActivations; hidAct++)
      {
         for (int outAct = 0; outAct < numOutputActivations; outAct++)
         {
            w.write(String.valueOf(hiddenToOutputWeights[hidAct][outAct]) + "\n");
         }
      }

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
      }

      for (int inAct = 0; inAct < numInputActivations; inAct++)
      {
         for (int hidAct = 0; hidAct < numHiddenActivations; hidAct++)
         {
            inputToHiddenWeights[inAct][hidAct] = Double.parseDouble(reader.readLine());
         }
      }

      for (int hidAct = 0; hidAct < numHiddenActivations; hidAct++)
      {
         for (int outAct = 0; outAct < numOutputActivations; outAct++)
         {
            hiddenToOutputWeights[hidAct][outAct] = Double.parseDouble(reader.readLine());
         }
      }
      r.close();
      reader.close();

   } // public static void loadWeights() throws NumberFormatException, IOException


} // public class Perceptron
