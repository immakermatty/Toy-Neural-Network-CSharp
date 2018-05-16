using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Xml.Serialization;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;

namespace JAX_Evolution
{

    [Serializable()]
    class NeuralNetwork : ISerializable
    {
        readonly Matrix.DataFunction Sigmoid = (e, i ,j) => 1 / (1 + Math.Exp(-e));
        readonly Matrix.DataFunction SigmoidDer = (e, i, j) => e * (1 - e);

        public int Seed { get; set; }
        Random RandGen { get; set; }
        public double LearningRate { get; set; }

        int[] Structure; //0...in, 1...hid0, last...out
        Matrix[] Bias; //0...hid0, last...out
        Matrix[] Weights; //0...in-hid0, last...hidn-out 
       

        public int Iterations { get; private set; }

        //original structure would be: int[] {a, b, c}
        public NeuralNetwork(int[] structure)
        {
            Structure = structure;
            Weights = new Matrix[structure.Length - 1];
            Bias = new Matrix[structure.Length - 1];

            for (int i = 0; i < Weights.Length; i++)
                Weights[i] = new Matrix(Structure[i+1], Structure[i]);

            for (int i = 0; i < Weights.Length; i++)
              Bias[i] = new Matrix(Structure[i+1], 1);
            
            Seed = new Random().Next();
            RandGen = new Random(Seed);

            for (int i = 0; i < Weights.Length; i++)
                Weights[i].Randomize(RandGen);

            for (int i = 0; i < Weights.Length; i++)
                Bias[i].Randomize(RandGen);

            LearningRate = 0.1;
        }

        //public NeuralNetwork(int a, int b, int c) //original
        //{
        //    //i,h,o layer node count
        //    Structure = new int[] { a, b, c };
        //    //Weights init
        //    Weights = new Matrix[2];
        //    Bias = new Matrix[2];

        //    //ih weights
        //    Weights[0] = new Matrix(Structure[1], Structure[0]);
        //    //ho weights
        //    Weights[1] = new Matrix(Structure[2], Structure[1]);

        //    //h bias
        //    Bias[0] = new Matrix(Structure[1], 1);
        //    //o bias
        //    Bias[1] = new Matrix(Structure[2], 1);

        //    //random Random object
        //    Seed = new Random().Next();
        //    RandGen = new Random(Seed);

        //    //randomize everything
        //    Weights[0].Randomize(RandGen);
        //    Weights[1].Randomize(RandGen);
        //    Bias[0].Randomize(RandGen);
        //    Bias[1].Randomize(RandGen);

        //    //Set learning rate of 0.5%
        //    SetLearningRate(0.005);
        //}

        public NeuralNetwork(NeuralNetwork nn)
        {
            Structure = nn.Structure;
            Bias = nn.Bias;
            Weights = nn.Weights;
            LearningRate = nn.LearningRate;
            Iterations = nn.Iterations;
            RandGen = nn.RandGen;
            Seed = nn.Seed;
        }

        //Seriazible stuff
        public NeuralNetwork(SerializationInfo info, StreamingContext context)
        {
            Structure = (int[])info.GetValue("Structure", typeof(int[]));
            Bias = (Matrix[])info.GetValue("Bias", typeof(Matrix[]));
            Weights = (Matrix[])info.GetValue("Weights", typeof(Matrix[]));
            LearningRate = (double)info.GetValue("LearningRate", typeof(double));
            Iterations = (int)info.GetValue("Iterations", typeof(int));
            RandGen = (Random)info.GetValue("RandGen", typeof(Random));
            Seed = (int)info.GetValue("Seed", typeof(int));
        }
     
        public void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            info.AddValue("Structure", Structure);
            info.AddValue("Bias", Bias);
            info.AddValue("Weights", Weights);
            info.AddValue("LearningRate", LearningRate);
            info.AddValue("Iterations", Iterations);
            info.AddValue("RandGen", RandGen);
            info.AddValue("Seed", Seed);
        }
     
        //public double[] Predict(double[] inputs_array)    //original
        //{         
        //    Matrix inputs = Matrix.FromArray(inputs_array);

        //    // Generating the Hidden Outputs
        //    Matrix hidden = Matrix.Multiply(Weights[0], inputs);
        //    hidden.Add(Bias[0]);
        //    // activation function!
        //    hidden.Map(Sigmoid);

        //    // Generating the Output's Output
        //    Matrix output = Matrix.Multiply(Weights[1], hidden);
        //    output.Add(Bias[1]);
        //    output.Map(Sigmoid);

        //    // Sending back to the caller
        //    return output.ToArray();
        //}

        public double[] Predict(double[] inputs_array)
        {
            Matrix matrix = Matrix.FromArray(inputs_array);

            for (int i = 0; i < Structure.Length - 1; i++)
            {
                // Generating the Hidden Outputs
                Matrix matrix_next = Matrix.Multiply(Weights[i], matrix);
                matrix_next.Add(Bias[i]);
                // activation function!
                matrix_next.Map(Sigmoid);
                //pass to another layer
                matrix = matrix_next;
            }

            // Sending last layer values to caller
            return matrix.ToArray();
        }

        //public void Train(double[] inputs_array, double[] outputs_array) //original
        //{
        //    if (Structure.Length != 3) throw new NotImplementedException();

        //    //CURRENT PREDICTION

        //    // Generating the Hidden Outputs
        //    Matrix inputs = Matrix.FromArray(inputs_array);
        //    Matrix hiddens = Matrix.Multiply(Weights[0], inputs);
        //    hiddens.Add(Bias[0]);
        //    // activation function!
        //    hiddens.Map(Sigmoid);

        //    // Generating the Output's Output
        //    Matrix outputs = Matrix.Multiply(Weights[1], hiddens);
        //    outputs.Add(Bias[1]);
        //    outputs.Map(Sigmoid);

        //    //ERROR CALCULATION

        //    //Desired output
        //    Matrix targets = Matrix.FromArray(outputs_array);

        //    // error = targets - outputs
        //    Matrix output_errors = Matrix.Substract(targets, outputs);

        //    //BACK PROPAGATION

        //    // gradient = outputs * (1 - outputs);
        //    // Calculate gradient
        //    Matrix gradients = Matrix.Map(outputs, SigmoidDer);
        //    gradients.MultiplyBy(output_errors);
        //    gradients.MultiplyBy(LearningRate);

        //    // Calculate deltas
        //    Matrix hidden_t = Matrix.Transpose(hiddens);
        //    Matrix weight_ho_deltas = Matrix.Multiply(gradients, hidden_t);

        //    // Adjust the weights by deltas
        //    Weights[1].Add(weight_ho_deltas);
        //    // Adjust the bias by its deltas (which is just the gradients)
        //    Bias[1].Add(gradients);

        //    // Calculate the hidden layer errors
        //    Matrix weight_ho_t = Matrix.Transpose(Weights[1]);
        //    Matrix hidden_errors = Matrix.Multiply(weight_ho_t, output_errors);

        //    // Calculate hidden gradient
        //    Matrix hidden_gradient = Matrix.Map(hiddens, SigmoidDer);
        //    hidden_gradient.MultiplyBy(hidden_errors);
        //    hidden_gradient.MultiplyBy(LearningRate);

        //    // Calcuate input->hidden deltas
        //    Matrix inputs_t = Matrix.Transpose(inputs);
        //    Matrix weight_ih_deltas = Matrix.Multiply(hidden_gradient, inputs_t);

        //    Weights[0].Add(weight_ih_deltas);
        //    // Adjust the bias by its deltas (which is just the gradients)
        //    Bias[0].Add(hidden_gradient);

        //    Iterations++;
        //}

        public void Train(double[] inputs_array, double[] outputs_array)
        {

            //CURRENT PREDICTION

            Matrix[] Layers = new Matrix[Structure.Length];

            // Generating the Hidden Outputs
            Layers[0] = Matrix.FromArray(inputs_array);

            for (int i = 1; i < Structure.Length; i++)
            {
                Layers[i] = Matrix.Multiply(Weights[i - 1], Layers[i - 1]);
                Layers[i].Add(Bias[i - 1]);
                Layers[i].Map(Sigmoid);
            }

            //ERROR CALCULATION

            Matrix error = Matrix.Substract(Matrix.FromArray(outputs_array), Layers.Last());

            //BACK PROPAGATION

            for (int i = Structure.Length - 1; i >= 1; i--)
            {         
                // gradient = outputs * (1 - outputs);
                // Calculate gradient
                Matrix gradients = Matrix.Map(Layers[i], SigmoidDer);
                gradients.MultiplyBy(error);
                gradients.MultiplyBy(LearningRate);

                // Calculate deltas
                Matrix hidden_t = Matrix.Transpose(Layers[i-1]);
                Matrix weight_ho_deltas = Matrix.Multiply(gradients, hidden_t);

                // Adjust the weights by deltas
                Weights[i - 1].Add(weight_ho_deltas);
                // Adjust the bias by its deltas (which is just the gradients)
                Bias[i - 1].Add(gradients);

                // Calculate the layer errors
                if (i > 1) error = Matrix.Multiply(Matrix.Transpose(Weights[i - 1]), error);
            }

            Iterations++;
        }

        public void Serelize(string filename)
        {
            //Binary
            Stream stream = File.Open(filename, FileMode.Create);
            BinaryFormatter bf = new BinaryFormatter();
            bf.Serialize(stream, this);
            stream.Close();

            ////Xml (does not work with multidim. arrays [,])
            //XmlSerializer seriliazer = new XmlSerializer(typeof(NeuralNetwork));
            //using (TextWriter tw = new StreamWriter(filename))
            //    seriliazer.Serialize(tw, this);
        }

        public static NeuralNetwork Deserelize(string filename)
        {
            //Binary
            Stream stream = File.Open(filename, FileMode.Open);
            BinaryFormatter bf = new BinaryFormatter();
            NeuralNetwork nn = (NeuralNetwork)bf.Deserialize(stream);
            stream.Close();
            return nn;

            ////Xml (does not work with multidim. arrays [,])
            //XmlSerializer deseriliazer = new XmlSerializer(typeof(NeuralNetwork));
            //TextReader tr = new StreamReader(filename);
            //return(NeuralNetwork) deseriliazer.Deserialize(tr);
        }

        // Adding function for neuro-evolution
        public NeuralNetwork Copy()
        {
            return new NeuralNetwork(this);
        }

        public delegate double MutationFuction(double x, Random r); 

        // Accept an arbitrary function for mutation
        public void Mutate(MutationFuction f)
        {
            Matrix.DataFunction g = (e, i, j) => f(e, RandGen);

            for (int i = 0; i < Weights.Length; i++)
                Weights[i].Map(g);
            for (int i = 0; i < Bias.Length; i++)
                Bias[i].Map(g);

        }

    }
 
    [Serializable()]
    class Matrix  //2d matrix math
    {
        double[,] Data { get; set; }
        int Rows { get; set; }
        int Cols { get; set; }

        public Matrix(int rows, int cols)
        {
            Rows = rows;
            Cols = cols;

            Data = new double[Rows, Cols];
            Map((x, i, j) => 0);
        }

        public Matrix Randomize(Random r)
        { 
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)         
                    Data[i, j] = r.NextDouble() * 2 - 1;

            return this;
        }

        public delegate double DataFunction(double e, int i, int j);

        public Matrix Map(DataFunction func)
        {
            // Apply a function to every element of matrix data
            for (int r = 0; r < Rows; r++)
            {
                for (int c = 0; c < Cols; c++)
                {
                    double val = Data[r,c];
                    Data[r,c] = func(val, r, c);
                }
            }
            return this;
        }

        public static Matrix Map(Matrix matrix, DataFunction func)
        {
            // Apply a function to every element of matrix
            return new Matrix(matrix.Rows, matrix.Cols).Map((e, i, j) => func(matrix.Data[i,j], i, j));
        }

        public static Matrix Multiply(Matrix a, Matrix b)
        {
            // Matrix product
            if (a.Cols != b.Rows) throw new Exception("Columns of A must match rows of B.");

            return new Matrix(a.Rows, b.Cols).Map(
                (e, i, j) =>
                    {
                        // Dot product of values in col
                        double sum = 0;
                        for (int k = 0; k < a.Cols; k++)
                        {
                            sum += a.Data[i, k] * b.Data[k, j];
                        }
                        return sum;
                    }
                );
        }

        //Vynásobí prvky Matrixů mezi sebou
        public Matrix MultiplyBy(Matrix matrix)
        {
            if (Rows != matrix.Rows || Cols != matrix.Cols)
                throw new Exception("Columns and Rows of A must match Columns and Rows of B.");
             
            // hadamard product
            return Map((e, i, j) => e * matrix.Data[i,j]);
        }

        //Vynásobí prvky matrixu konstatnou
        public Matrix MultiplyBy(double n)
        {
            // hadamard product
            return Map((e, i, j) => e * n);
        }

        public static Matrix Transpose(Matrix matrix)
        {
            return new Matrix(matrix.Cols, matrix.Rows).Map((e, i, j) => matrix.Data[j,i]);
        }

        public static Matrix FromArray(double[] inputs_array)
        {
            //One line copy array to Matrix object
            return new Matrix(inputs_array.Length, 1).Map((e, i, j) => inputs_array[i]);
        }

        public double[] ToArray()
        {
            List<double> list = new List<double>();
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)           
                    list.Add(Data[i,j]);
                         
            return list.ToArray();
        }

        public Matrix Add(Matrix matrix)
        {
            if (Rows != matrix.Rows || Cols != matrix.Cols)
                throw new Exception("Columns and Rows of A must match Columns and Rows of B.");

            return Map((e, i, j) => e + matrix.Data[i,j]);
        }

        public Matrix Add(double a)
        {
            return this.Map((e, i, j) => e + a);
        }

        public static Matrix Substract(Matrix a, Matrix b)
        {
            if (a.Rows != b.Rows || a.Cols != b.Cols)
                throw new Exception("Columns and Rows of A must match Columns and Rows of B.");

            // Return a new Matrix a-b
            return new Matrix(a.Rows, a.Cols).Map((e, i, j) => a.Data[i, j] - b.Data[i, j]);
        }

       

    }
}
