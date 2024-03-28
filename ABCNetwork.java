import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.Math;
/**
 * @author Saanvi Bhargava
 * Created: January 27, 2024
 * 
 * An A-B-C neural network that has one hidden layer. 
 * The hidden layer, input layer, and output layer can have any number of activations.
 * Computes an output based on initial input activations, configured in the code itself, along with the weights between layers.
 * The network trains by simple gradient descent and uses the sigmoid activation function. 
 * The computed output is then compared to the expected output using an error function. 
 * After the network is run on all training sets, the net and mean error is calculated. 
 */
public class ABCNetwork
{
   static double lambda;                     // the learning factor used to control the magnitude of weight changes
   static boolean training;                  // true if the network is in training mode, false if the network is in running mode
   static int numInputActivations;           // number of input activations
   static int numHiddenActivations;          // number of hidden activations in the singular hidden layer
   static int numOutputActivations;          // number of output activations
   static double lowerrand;                  // minimum randomized weight value
   static double higherrand;                 // maximum randomized weight value
   static int maxIterations;                 // maximum number of iterations allowed before training is stopped
   static boolean randomize;                 // true if weights are to be randomized, false if weights are to be predetermined
   static double errorThreshold;             // maximum error tolerance for training completion
   static double errorReached;               // minimum error achieved at completion of training
   static int numTestCases;                  // the number of training sets 
   static boolean applySigmoid;              // true if sigmoid function is to be used for the activation function
   static double inputActivations[];         // the input activations in order as defined by the low-level design document
   static double hiddenActivations[];        // the hidden activations in order as defined by the low-level design document
   static double inputToHiddenWeights[][];   // weights array for the calculations from input to hidden activations
   static double hiddenToOutputWeights[][];  // weights array for the calculations from hidden to output activations 
   static double truthTableInputs[][];       // truth table with each test case and the input activations
   static double truthTableExpOutputs[][];   // truth table with the expected output activations
   static double truthTableActOutputs[][];   // truth table with the actual output activations
   static int numInterations;                // number of iterations training took to reach the error threshold
   static double inToHidWeightChange[][];    // the array of the change in weights for input to hidden activations
   static double hidToOutWeightChange[][];   // the array of the change in weights for hidden to output activations
   static double omegai[];                   // omegai array to assist with weight changes as defined by the design document
   static double psii[];                     // psii array to assist with weight changes as defined by the design documents
   static double omegaj[];                   // omegaj array to assist with weight changes as defined by the design document
   static double psij[];                     // psij array to assist with weight changes as defined by the design document
   static String weightsFile;                // file name for weights
   
   public static void main(String[] args) throws IOException 
   {
      setConfigParams();      // configures network parameters
      echoConfigParams();     // prints network parameters and checks for issues
      allocateArrMem();       // configures space for the arrays to be used for the network
      populateArrays();       // populates weights arrays and truth table
      
      if (training)
      {
         train();             // trains the network to converge to a maximum error or number of iterations
         reportResults();     // reports training results
         saveWeights();
      }
      loadWeights();       
      run();                  // runs the network on the test cases
      reportResults();        // reports expected output from running results
   } // public static void main(String[] args) 

/**
 * Initializes all the instance variables, excluding arrays.
 */
   public static void setConfigParams()
   {
      lambda = 0.3;           
      training = true;
      numInputActivations = 2;
      numHiddenActivations = 1;
      numOutputActivations = 3;
      lowerrand = 0.1;
      higherrand = 1.5;
      maxIterations = 100000;
      errorThreshold = 0.0002;
      numTestCases = 4;
      applySigmoid = true;
      randomize = true;
      weightsFile = "weights.txt";
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

         if (randomize)
         {
            System.out.println("The network is starting with random weights between " + lowerrand + " and " + higherrand + ".");
         }
         else 
         {
            System.out.println("The network is starting training with predetermined weights.");
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
         inToHidWeightChange = new double[numInputActivations][numHiddenActivations];
         hidToOutWeightChange = new double[numHiddenActivations][numOutputActivations];
         omegai = new double[numOutputActivations];
         psii = new double[numOutputActivations];
         omegaj = new double[numHiddenActivations];
         psij = new double[numHiddenActivations];
      }
      
   } // public static void allocateArrMem()

/**
 * Populating the weight arrays and truth table with initial values.
 * Weight array values depends on randomization and truth table is currently manually entered.
 */
   public static void populateArrays()
   {  
/**
 * populating weights
 */ 
      if (randomize)
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
               hiddenToOutputWeights[hidAct][outAct] = genRandWeight();            // weights for hidden to output calculations
            }
            
         }
      } // if (randomize)
      else
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

/**
 * Manually populating the truth table. This is currently populated for 2 input activations.
 */
      truthTableInputs[0][0] = 0;
      truthTableInputs[0][1] = 0;
      // truthTableInputs[0][2] = 0;      // for testing A > 2
      

      truthTableExpOutputs[0][0] = 0;     // AND
      truthTableExpOutputs[0][1] = 0;     // OR
      truthTableExpOutputs[0][2] = 0;     // XOR
      
      truthTableInputs[1][0] = 0;
      truthTableInputs[1][1] = 1;
      // truthTableInputs[1][2] = 0;      // for testing A > 2
      

      truthTableExpOutputs[1][0] = 0;     // AND
      truthTableExpOutputs[1][1] = 1;     // OR
      truthTableExpOutputs[1][2] = 1;     // XOR

      truthTableInputs[2][0] = 1;
      truthTableInputs[2][1] = 0;
      // truthTableInputs[2][2] = 0;      // for testing A > 2
      

      truthTableExpOutputs[2][0] = 0;     // AND
      truthTableExpOutputs[2][1] = 1;     // OR
      truthTableExpOutputs[2][2] = 1;     // XOR

      truthTableInputs[3][0] = 1;
      truthTableInputs[3][1] = 1;
      // truthTableInputs[3][2] = 0;      // for testing A > 2
      

      truthTableExpOutputs[3][0] = 1;     // AND
      truthTableExpOutputs[3][1] = 1;     // OR
      truthTableExpOutputs[3][2] = 0;     // XOR     
 

   } // public static void populateArrays()

/**
 * This method is responsible for training the perceptron by updating its weights to minimize the error.
 * Calculates the value of the activations in the hidden layers using dot products of the weights of the input layers.
 * It employs the perceptron learning algorithm, 
 * iterating through the test cases until convergence or reaching a maximum number of iterations.
 * The convergence is determined by the error falling below a specified threshold.
 */
   public static void train()
   {
      double avgErr = 1.0;
      int iter = 0;
      boolean done = false;
      double weightChange = 0.0;
      double E = 0.0;
      double diff = 0.0;

      System.out.println("Beginning training.\n");

      while (!done) // looping until end condition is met (error is below threshold or max iterations is reached)
      {
         double totalErr = 0.0;

         for (int testCase = 0; testCase < numTestCases; testCase++) // in each iteration, looping through all training cases
         {
            
            runTestCase(testCase); // running network to find Fi for later weight changes

            for (int i = 0; i < numOutputActivations; i++)
            {
               omegai[i] = truthTableExpOutputs[testCase][i] - truthTableActOutputs[testCase][i];
               psii[i] = omegai[i] * derivSigmoid(truthTableActOutputs[testCase][i]);
            }

/**
 * finding weight change and putting in change array
 */
            for (int outAct = 0; outAct < numOutputActivations; outAct++)
            {
               for (int hidAct = 0; hidAct < numHiddenActivations; hidAct++)
               {
                  weightChange = lambda * hiddenActivations[hidAct]*psii[outAct];
                  hidToOutWeightChange[hidAct][outAct] = weightChange;
               }
            }
            
/**
 * doing weight changes for input to hidden weights
 */
            for (int hidAct = 0; hidAct < numHiddenActivations; hidAct++) 
            {
               omegaj[hidAct] = 0.0;
               for (int outAct = 0; outAct < numOutputActivations; outAct++)
               {
                  omegaj[hidAct] += psii[outAct] * hiddenToOutputWeights[hidAct][outAct];
               }
               psij[hidAct] = omegaj[hidAct] * derivSigmoid(hiddenActivations[hidAct]);

               for (int inAct = 0; inAct < numInputActivations; inAct++)
               {
                  weightChange = lambda*inputActivations[inAct]*psij[hidAct];
                  inToHidWeightChange[inAct][hidAct] = weightChange;
               }
            } // for (int hidAct = 0; hidAct < numHiddenActivations; hidAct++)

/**
 * determining new input activation to hidden activation weights based on weight changes
 */
            for (int inAct = 0; inAct < numInputActivations; inAct++)
            {
               for (int hidAct = 0; hidAct < numHiddenActivations; hidAct++)
               {
                  inputToHiddenWeights[inAct][hidAct] += inToHidWeightChange[inAct][hidAct];
               }
            }

/**
 * determining new hidden activation to output activation weights based on weight changes
 */
            for (int hidAct = 0; hidAct < numHiddenActivations; hidAct++)
            {
               for (int outAct = 0; outAct < numOutputActivations; outAct++)
               {
                  hiddenToOutputWeights[hidAct][outAct] += hidToOutWeightChange[hidAct][outAct];
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
         
         iter++;
         avgErr = (totalErr/numTestCases);
         done = (avgErr < errorThreshold || iter >= maxIterations); // defining done
      } // while (!done)
      
      errorReached = avgErr; // final error achieved below minimum error threshold or because of completion of iterations
      numInterations = iter;
      return;
   } // public static void train()

/**
 * If sigmoid is to be applied, this method applies sigmoid function to activation for weight calculation.
 * The sigmoid function is as follows: f(x) = 1/(1+e^(-x))
 */
   public static double sigmoid(double theta)
   {
      if (applySigmoid)
      {
         double efunc = Math.exp((-theta));
         theta = 1.0/(1.0 + efunc);
      }
      
      return theta;
   } // public static double sigmoid(double theta)

/**
 * Calculates the value of the derivative of the sigmoid function for F. 
 */
   public static double derivSigmoid(double F)
   {
      return F * (1.0 - F);
   }

/**
 * Runs the network on all test cases in the truth table and updates the truth table with the determined output.
 */
   public static void run()
   {
      System.out.println("\nBeginning predictions.");

      for (int testCase = 0; testCase < numTestCases; testCase++)
      {
         runTestCase(testCase);
      }
   } // public static void run()

/**
 * Runs the network on a singular test case. Calculates weights and then F0.
 */
   public static void runTestCase(int testCaseNum)
   {
      double thetaj = 0.0;
      double thetai = 0.0;
      
/**
 * repopulates input activations based on truth table
 */
      for (int inAct = 0; inAct < numInputActivations; inAct++)
      {
         inputActivations[inAct] = truthTableInputs[testCaseNum][inAct];
      }

/**
 * calculates hidden activations based on weights and input activation
 */
      for (int hidAct = 0; hidAct < numHiddenActivations; hidAct++)
      {
         thetaj = 0.0;
         for (int inAct = 0; inAct < numInputActivations; inAct++)
         {
            thetaj += inputToHiddenWeights[inAct][hidAct]*inputActivations[inAct]; // theta = weight*inputAct for each input act
         }

         hiddenActivations[hidAct] = sigmoid(thetaj);
      } // for (int hidAct = 0; hidAct < numHiddenActivations; hidAct++)
      
/**
 * determining F0 based on hidden activations and weights
 */
      for (int outAct = 0; outAct < numOutputActivations; outAct++)
      {
         thetai = 0.0;
         for (int hidAct = 0; hidAct < numHiddenActivations; hidAct++)
         {
            thetai += hiddenToOutputWeights[hidAct][outAct]*hiddenActivations[hidAct];
         }

         truthTableActOutputs[testCaseNum][outAct] = sigmoid(thetai);
      }
      
   } // public static double runTestCase(int testCaseNum)

/**
 * Prints the entire truth table.
 */
   public static void printTruthTable()
   {
      System.out.println("Truth table:\n");

      String headerStr = "Test Case #\tInputs";

      for (int numInputs = 0; numInputs < numInputActivations; numInputs++)
      {
         headerStr += "\t";
      }

      headerStr += "AND \t\t\t OR \t\t\t XOR";

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
      System.out.println("Truth table:\n");

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
      if (training)
      {
         System.out.println("Training is completed. Training results are:");
         System.out.println("\nThe final error was " + errorReached + ".");

         if (errorReached < errorThreshold)
         {
            System.out.println("Training terminated since error threshold was reached after " + numInterations + " iterations.");
         }
         else 
         {
            System.out.println("Training terminated because it completed max iterations.");
         }
      } // if (training)
      else 
      {
         System.out.println("\nFinished running the network. Predictions are:\n");
         printExpTruthTable();
         printTruthTable();
      }
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
 * @throws IOException 
 * @throws NumberFormatException 
 */
   public static void loadWeights() throws NumberFormatException, IOException
   {
      FileReader r = new FileReader(weightsFile);
      BufferedReader reader = new BufferedReader(r);

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

      reader.close();
   } // public static void loadWeights() throws NumberFormatException, IOException

} // public class Perceptron
