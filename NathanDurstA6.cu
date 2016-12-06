//**********************************
//Nathan Durst
//FFT Cuda Program
//December, 5 2016
//**********************************
//This application uses cuda c and implements
// the Cooley-Tukey FFT algorithm to transforms 
// an array of complex numbers into a data set
// correlation of complex numbers.
#include <stdio.h>
#include <math.h>
#define N 16384
#define PI 3.14

//kernel function declaration
__global__ void FFT(float * R, float * I, float * xR, float * xI);

int main()
{
	float R[N] = {0};
	float I[N] = {0};
	float xR[N], xI[N], *Rd, *Id, *xRd, *xId, elapsed;
	int i, size = N * sizeof(int);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(stop);
	cudaEventRecord(start, 0);
	
	//initialize arrays of real and imaginary numbers
	R[0] = 3.6; R[1] = 2.9; R[2] = 5.6; R[3] = 4.8;
	R[4] = 3.3; R[5] = 5.9; R[6] = 5.0; R[7] = 4.3;
	I[0] = 2.6; I[1] = 6.3; I[2] = 4.0; I[3] = 9.1;
	I[4] = 0.4; I[5] = 4.8; I[6] = 2.6; I[7] = 4.1;
	
	//allocate size of arrays on device and store them in 
	// specified array variable names
	cudaMalloc((void**)&Rd, size);
	cudaMalloc((void**)&Id, size);
	cudaMalloc((void**)&xRd, size);
	cudaMalloc((void**)&xId, size);
	
	//copy initialized arrays to arrays on device
	cudaMemcpy(Rd, R, size, cudaMemcpyHostToDevice);
	cudaMemcpy(Id, I, size, cudaMemcpyHostToDevice);
	
	//determine dimensions of block and threads used by the kernel
	dim3 dimGrid((N/1024),1);
	dim3 dimBlock(1024, 1);

	//call kernel function FFT
	FFT<<<dimGrid, dimBlock>>>(Rd, Id, xRd, xId);
	
	//copy results from device arrays to the host arrays
	cudaMemcpy(xR, xRd, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(xI, xId, size, cudaMemcpyDeviceToHost);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	//print results
	for (i = 0; i < 8; i++)
		printf("X[%d]: %0.1f + %0.1fi\n", i, xR[i], xI[i]);
	printf("The elapsed time of the program was %.2f ms\n", elapsed);
		
	//free space on device
	cudaFree(Rd);
	cudaFree(Id);
	cudaFree(xRd);
	cudaFree(xId);
}

__global__ void FFT(float * R, float * I, float * xR, float * xI)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	float real = 0, imag = 0;
	
	//iterate through entire array for each index and calculate even
	// and odd for real and imaginary numbers.
	for (int i = 0; i<(N/2); i++)
	{
		//even
		real += R[i] * cos((2*PI*(i*2))/N) - I[i] * sin((2*PI*id*(i*2))/N);
		imag += R[i] * -sin((2*PI*(i*2))/N) + I[i] * cos((2*PI*id*(i*2))/N);
		
		//odd
		real += R[i] * cos((2*PI*(i*2+1))/N) - I[i] * sin((2*PI*id*(i*2+1))/N);
		imag += R[i] * -sin((2*PI*(i*2+1))/N) + I[i] * cos((2*PI*id*(i*2+1))/N);
	}
	xR[id] = real;
	xI[id] = imag;
}