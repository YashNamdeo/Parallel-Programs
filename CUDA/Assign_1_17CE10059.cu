#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void process_kernel1(float *input1, float *input2, float *output, int datasize){
        
    int blockNum = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
    int threadNum = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * (blockDim.x) + threadIdx.x;
    int i = blockNum * (blockDim.x * blockDim.y * blockDim.z) + threadNum;    
    if (i < datasize){
        output[i]=sin(input1[i])+cos(input2[i]);
    }
}

__global__ void process_kernel2(float *input, float *output, int datasize){
    
    int blockNum = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
    int threadNum = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * (blockDim.x) + threadIdx.x;
    int i = blockNum * (blockDim.x * blockDim.y * blockDim.z) + threadNum;
    if (i < datasize){
        output[i]=log(input[i]);
    }
}

__global__ void process_kernel3(float *input, float *output, int datasize){
    int blockNum = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
    int threadNum = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * (blockDim.x) + threadIdx.x;
    int i = blockNum * (blockDim.x * blockDim.y * blockDim.z) + threadNum;
    if (i < datasize){
        output[i]=sqrt(input[i]);
    }
}

int main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    int numElements = 0;
    scanf("%d",&numElements);
    
    size_t size = numElements * sizeof(float);

    
    float *h_input1 = (float *)malloc(size);
    float *h_input2 = (float *)malloc(size);
    float *h_output = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_input1 == NULL || h_input2 == NULL || h_output == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    //Trying for random values 
    /*for (int i = 0; i < numElements; ++i)
    {
        h_input1[i] = rand()/(float)RAND_MAX;
        h_input2[i] = rand()/(float)RAND_MAX;
    }*/
    
    //taking inputs
    printf("Enter input1 elements: \n");
    for (int i = 0; i < numElements; ++i)
    scanf("%f", &h_input1[i]);

    printf("Enter input2 elements: \n");
    for (int i = 0; i < numElements; ++i)
    scanf("%f", &h_input2[i]);

    float *d_input1 = NULL;
    err = cudaMalloc((void **)&d_input1, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    float *d_input2 = NULL;
    err = cudaMalloc((void **)&d_input2, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *d_output1 = NULL;
    err = cudaMalloc((void **)&d_output1, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
 

    float *d_output2 = NULL;
    err = cudaMalloc((void **)&d_output2, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
 

    float *d_output3 = NULL;
    err = cudaMalloc((void **)&d_output3, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors input1 and input2 in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_input1, h_input1, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector input1 from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_input2, h_input2, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector input2 from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    dim3 threadsPerBlock1(32,32,1);
    dim3 blocksPerGrid1(4,2,2);
    process_kernel1<<<blocksPerGrid1, threadsPerBlock1>>>(d_input1, d_input2, d_output1, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch process_kernel1 kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
 
    dim3 threadsPerBlock2(8,8,16);
    dim3 blocksPerGrid2(2,8,1);
    process_kernel2<<<blocksPerGrid2, threadsPerBlock2>>>(d_output1, d_output2, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch process_kernel2 kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
 
    dim3 threadsPerBlock3(128,8,1);
    dim3 blocksPerGrid3(16,1,1);
    process_kernel3<<<blocksPerGrid3, threadsPerBlock3>>>(d_output2, d_output3, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch process_kernel3 kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_output, d_output3, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector output from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        printf("%.2f ",h_output[i]);
    }


    // Free device global memory
    err = cudaFree(d_input1);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector input1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_input2);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector input2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_output1);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector output1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
 
    err = cudaFree(d_output2);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector output2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
 
    err = cudaFree(d_output3);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector output3 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_input1);
    free(h_input2);
    free(h_output);

   
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return 0;
}