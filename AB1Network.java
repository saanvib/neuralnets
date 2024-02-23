import java.lang.Math;
/**
 * @author Saanvi Bhargava
 * Created: January 27, 2024
 * 
 * An A-B-1 neural network that has one hidden layer. 
 * The hidden layer and input layer can have any number of activations, and the output layer can have only 1 activation.
 * Computes an output based on initial input activations, configured in the code itself, along with the weights between layers.
 * The network trains by simple gradient descent and uses the sigmoid activation function. 
 * The computed output is then compared to the expected output using an error function. 
 * After the network is run on all training sets, the net and mean error is calculated. 
 */
public class AB1Network
{
   static double lambda;                     // the learning factor used to control the magnitude of weight changes
   static boolean training;                  // true if the network is in training mode, false if the network is in running mode
   static int numInputActivations;           // number of input activations
   static int numHiddenActivations;          // number of hidden activations in the singular hidden layer
   static int numOutputActivations;          // number of output activations (always 1 as of now)
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
   static double hiddenToOutputWeights[];    // weights array for the calculations from hidden to output activations 
   static double truthTable[][];             // truth table with each test case and the expected and actual output
   static int numInterations;                // number of iterations training took to reach the error threshold
   static double inToHidWeightChange[][];    // the array of the change in weights for input to hidden activations
   static double hidToOutWeightChange[];     // the array of the change in weights for hidden to output activations
   static double omegaArr[];                 // array to assist with determining weight changes
   static double psiArr[];                   // array to assist with determining weight changes
   
   public static void main(String[] args) 
   {
      setConfigParams();      // configures network parameters
      echoConfigParams();     // prints network parameters and checks for issues
      allocateArrMem();       // configures space for the arrays to be used for the network
      populateArrays();       // populates weights arrays and truth table
      
      if (training)
      {
         train();             // trains the network to converge to a maximum error or number of iterations
         reportResults();     // reports training results
      }

      training = false;       
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
      numHiddenActivations = 5;
      numOutputActivations = 1;
      lowerrand = -1.5;
      higherrand = 1.5;
      maxIterations = 100000;
      errorThreshold = 0.0002;
      numTestCases = 4;
      applySigmoid = true;
      randomize = true;
   } // public static void setConfigParams()

/**
 * Displays the key details of the network configuration such as
 * learning factor, dimensions, random ranges, etc
 */
   public static void echoConfigParams()
   {
      System.out.println("\n\nNetwork Configuration:\n");
      System.out.println("The configuration is a " + numInputActivations + "-" + numHiddenActivations + "-1 network.");

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
      truthTable = new double[numTestCases][numInputActivations+2]; 
      inputToHiddenWeights = new double[numInputActivations][numHiddenActivations];
      hiddenToOutputWeights = new double[numHiddenActivations];

      if (training)
      {
         inToHidWeightChange = new double[numInputActivations][numHiddenActivations];
         hidToOutWeightChange = new double[numHiddenActivations];
         omegaArr = new double[numHiddenActivations];
         psiArr = new double[numHiddenActivations];
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
            hiddenToOutputWeights[hidAct] = genRandWeight();            // weights for hidden to output calculations
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
         hiddenToOutputWeights[0] = 0.4;
         hiddenToOutputWeights[1] = 0.1;
      }

/**
 * Manually populating the truth table. This is currently populated for 2 input activations.
 */
      truthTable[0][0] = 0.0;
      truthTable[0][1] = 0.0;
      truthTable[0][2] = 0.0;    // change this value for changing expected output (test case 1)
      
      truthTable[1][0] = 0.0;
      truthTable[1][1] = 1.0;
      truthTable[1][2] = 1.0;    // change this value for changing expected output (test case 2)
      
      truthTable[2][0] = 1.0;
      truthTable[2][1] = 0.0;
      truthTable[2][2] = 1.0;    // change this value for changing expected output (test case 3)
      
      truthTable[3][0] = 1.0;
      truthTable[3][1] = 1.0;
      truthTable[3][2] = 0.0;    // change this value for changing expected output (test case 4)

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
      double F0 = 0.0;
      double omega = 0.0;
      double psi = 0.0;

      System.out.println("Beginning training.\n");

      while (!done) // looping until end condition is met (error is below threshold or max iterations is reached)
      {
         double totalErr = 0.0;

         for (int testCase = 0; testCase < numTestCases; testCase++) // in each iteration, looping through all training cases
         {
            F0 = runTestCase(testCase); // running network to find F0 for later weight changes
            omega = truthTable[testCase][numInputActivations]-F0;
            psi = omega*(derivSigmoid(F0));
            
            iter++;
/**
 * finding weight change and putting in change array
 */
            for (int hidAct = 0; hidAct < numHiddenActivations; hidAct++)
            {
               weightChange = lambda*hiddenActivations[hidAct]*psi;
               hidToOutWeightChange[hidAct] = weightChange;
            }

/**
 * doing weight changes for input to hidden weights
 */
            for (int hidAct = 0; hidAct < numHiddenActivations; hidAct++) 
            {
               omegaArr[hidAct] = psi*hiddenToOutputWeights[hidAct];
               psiArr[hidAct] = omegaArr[hidAct]*hiddenActivations[hidAct]*(1-hiddenActivations[hidAct]);
               for (int inAct = 0; inAct < numInputActivations; inAct++)
               {
                  weightChange = lambda*inputActivations[inAct]*psiArr[hidAct];
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
               hiddenToOutputWeights[hidAct] += hidToOutWeightChange[hidAct];
            }

            F0 = runTestCase(testCase); // running network again to determine error with new weights
            E = ((truthTable[testCase][numInputActivations]-F0)*(truthTable[testCase][numInputActivations]-F0)*0.5);
            totalErr += E; // adding to total error for all test cases in this iteration
            
         } // for (int testCase = 0; testCase < numTestCases; testCase++)

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
 * Calculates the value of the derivative of the sigmoid function at a parameter f. 
 */
   public static double derivSigmoid(double f)
   {
      double theta = sigmoid(f);
      return theta*(1.0 - theta);
   }

/**
 * Runs the network on all test cases in the truth table and updates the truth table with the determined output.
 */
   public static void run()
   {
      System.out.println("\nBeginning predictions.");

      for (int testCase = 0; testCase < numTestCases; testCase++)
      {
         truthTable[testCase][numInputActivations + 1] = runTestCase(testCase);
      }
   } // public static void run()

/**
 * Runs the network on a singular test case. Calculates weights and then F0.
 */
   public static double runTestCase(int testCaseNum)
   {
      double theta = 0.0;
      
/**
 * repopulates input activations based on truth table
 */
      for (int inAct = 0; inAct < numInputActivations; inAct++)
      {
         inputActivations[inAct] = truthTable[testCaseNum][inAct];
      }

/**
 * calculates hidden activations based on weights and input activation
 */
      for (int hidAct = 0; hidAct < numHiddenActivations; hidAct++)
      {
         theta = 0.0;
         for (int inAct = 0; inAct < numInputActivations; inAct++)
         {
            theta += inputToHiddenWeights[inAct][hidAct]*inputActivations[inAct]; // theta = weight*inputAct for each input act
         }

         hiddenActivations[hidAct] = sigmoid(theta);
      } // for (int hidAct = 0; hidAct < numHiddenActivations; hidAct++)
      
/**
 * determining F0 based on hidden activations and weights
 */
      theta = 0.0;
      for (int hidAct = 0; hidAct < numHiddenActivations; hidAct++)
      {
         theta += hiddenToOutputWeights[hidAct]*hiddenActivations[hidAct];
      }

      return sigmoid(theta);
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

      headerStr += "Expected Output\tActual Output";
      System.out.println(headerStr);

      for (int t = 0; t < numTestCases; t++)
      {
         String printStr = "Test Case " + (t+1) + ": \t";

         for (int numIn = 0; numIn < numInputActivations; numIn++)
         {
            printStr += truthTable[t][numIn] + "\t";
         }

         printStr += truthTable[t][numInputActivations] + "\t\t" + truthTable[t][numInputActivations+1];
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
         
         printTruthTable();
      }
      training = false;
   } // public static void reportResults()

/**
 * Generates random weight between a lower and upper bound (inclusive of the lower bound but exclusive of the upper bound).
 */
   public static double genRandWeight()
   {
      return ((higherrand - lowerrand) * Math.random()) + lowerrand;
   } // public static double genRandWeight()
} // public class Perceptron
