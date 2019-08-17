#include <math.h>
#define NIN (2+1) // number of input neurons
#define NHID (4+1) // number of hidden neurons
#define NOUT 1 // number of output neurons
float w_in [NIN][NHID]; // in weights from 3 to 4 neur.
 float w_out[NHID][NOUT]; // out weights from 4 to 1 neur.

 float sigmoid(float x)
{ return 1.0 / (1.0 + exp(-x));
}
 void feedforward(float N_in[NIN], float N_hid[NHID],
float N_out[NOUT])
 { int i,j;
 // calculate activation of hidden neurons
 N_in[NIN-1] = 1.0; // set bias input neuron
 for (i=0; i<NHID-1; i++)
 { N_hid[i] = 0.0;
 for (j=0; j<NIN; j++)
 N_hid[i] += N_in[j] * w_in[j][i];
 N_hid[i] = sigmoid(N_hid[i]);
 }
 N_hid[NHID-1] = 1.0; // set bias hidden neuron
 // calculate activation and output of output neurons
for (i=0; i<NOUT; i++)
 { N_out[i] = 0.0;
 for (j=0; j<NHID; j++)
 N_out[i] += N_hid[j] * w_out[j][i];
 N_out[i] = sigmoid(N_out[i]);
}

}