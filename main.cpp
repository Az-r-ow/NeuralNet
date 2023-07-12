#include "main.hpp"
#include "src/Neuron.hpp"

void initializeSrand() {
   struct timeval time; 
   gettimeofday(&time,NULL);
   srand((time.tv_sec * 1000) + (time.tv_usec / 1000));
}

int main(int argc, char *argv[])
{
   initializeSrand();
   // Create the Perceptron.
   Neuron p = Neuron(3);
   // The input is 3 values: x,y and bias.
   float point[] = {50,-12,1};
   float point2[] = {90,12,1};
   // The answer

   MatrixXd m(28,28);
   m(0,0) = 3;
   m(1,0) = 2.5;
   m(0,1) = -1;
   m(1,1) = m(1,0) + m(0,1);
   std::cout << m << std::endl;

   return 0; 
}