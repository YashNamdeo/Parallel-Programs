#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define A(i,j) A[(i)*cols+(j)]  // row-major layout
#define C(i,j) C[(i)*cols+(j)]  // row-major layout

__global__ void convolution(float *A, float *C,long long int N)
{
    //Filter
    int filter[3][3] = { { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 } };

    //Needs for row-major layout
    int cols = N + 2;
    //int i = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int threadBlockSize = (N+2)/ blockDim.x;//The amount of processing per thread

    for (int b = threadIdx.x * threadBlockSize; b < (threadIdx.x + 1) * threadBlockSize; b++){
        
        i = b;
        
        for (int j = 0; j < N + 1; j++){//columns
            
            if (0 < i && i < N + 1 && 0 < j && j < N + 1)
            {
                float value = 0;
                value = value + A(i - 1, j - 1) *  filter[0][0];
                value = value + A(i - 1, j)     *  filter[0][1];
                value = value + A(i - 1, j + 1) *  filter[0][2];
                value = value + A(i, j - 1)     *  filter[1][0];
                value = value + A(i, j)         *  filter[1][1];
                value = value + A(i, j + 1)     *  filter[1][2];
                value = value + A(i + 1, j - 1) *  filter[2][0];
                value = value + A(i + 1, j)     *  filter[2][1];
                value = value + A(i + 1, j + 1) *  filter[2][2];
                C(i, j) = value;
            }
        }
    }

}

int main(void)
{
    // Error code to check return values for CUDA calls
    int t;
    scanf("%d",&t);
    while(t--)
    {
        cudaError_t err = cudaSuccess;

        long long int N = 0;
        scanf("%lld ",&N);

        float A[N+2][N+2] = {};//+2 for padding matrix
        float *C = 0;
                int cols = N;
        float *A_d = 0, *C_d = 0;

        size_t memorySize = (N+2) * (N+2);
        
        // Verify that allocations succeeded
        /* if (h_input == NULL)
        {
            fprintf(stderr, "Failed to allocate host vectors!\n");
            exit(EXIT_FAILURE);
        }*/

        //Trying for random values 
        for (long long int i = 0; i < N+2; ++i)
            for (long long int j = 0; j < N+2; ++j)
        {
            A[i][j] = 0;
        }
                /*
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
        {
            A[i+1][j+1] = rand()/(float)RAND_MAX;
        }
        */
        //taking inputs
        
        printf("Enter input elements: \n");
        for (long long int i = 0; i < N; ++i)
            for (long long int j = 0; j < N; ++j)
                scanf("%f", &A[i+1][j+1]);
        

        //C = (int *)malloc(sizeof(*C)*memorySize);
        cudaMalloc((void**)&C, sizeof(*C)*memorySize);
        cudaMalloc((void**)&A_d, sizeof(*A_d)*memorySize);
        cudaMalloc((void**)&C_d, sizeof(*C_d)*memorySize);

        /*float *d_input = NULL;
        err = cudaMalloc((void **)&d_input, sizeof(*A_d) size);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Allocate the device output vector C
        float *d_output = NULL;
        err = cudaMalloc((void **)&d_output, size);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        */
        // Copy the host input vectors input1 and input2 in host memory to the device input vectors in
        // device memory
        printf("Copy input data from the host memory to the CUDA device\n");
        err = cudaMemcpy(A_d, A, sizeof(*A_d)*memorySize, cudaMemcpyHostToDevice);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector input1 from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Launch the Vector Add CUDA Kernel
        dim3 threadsPerBlock1(1,1,1);
        dim3 blocksPerGrid1(1024,1,1);
        convolution<<<blocksPerGrid1, threadsPerBlock1>>>(A_d, C_d, N);
        err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch process_kernel1 kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Copy the device result vector in device memory to the host result vector
        // in host memory.
        printf("Copy output data from the CUDA device to the host memory\n");
        err = cudaMemcpy(C, C_d, sizeof(*C)*memorySize, cudaMemcpyDeviceToHost);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector output from device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Verify that the result vector is correct
        printf("Results\n");
        for (long long int i = 0; i < N+2; ++i)
        { 
            for(long long int j =0; j < N+2; ++j)
            {
                printf("%.2f ",C(i,j));
            }
            printf("\n");
        }

        // Free device global memory
        err = cudaFree(A_d);;

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector input1 (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        
        err = cudaFree(C_d);;

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector output1 (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
     
        
        //free(C);

       
        err = cudaDeviceReset();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    return 0;
}
