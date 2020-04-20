#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_DIM  32
#define b_row  32

__global__ void transpose(float *output, float *input, int width, int height, int n)
{
    __shared__ float tile[TILE_DIM][TILE_DIM+1];

    int x_index = blockIdx.x * TILE_DIM + threadIdx.x;
    int y_index = blockIdx.y * TILE_DIM + threadIdx.y;
    int index_in = x_index + (y_index)*width;

    x_index = blockIdx.y * TILE_DIM + threadIdx.x;
    y_index = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = x_index + (y_index)*height;

    for (int r=0; r < n; r++)
    {
        for (int i=0; i<TILE_DIM; i+=b_row)
        {
            tile[threadIdx.y+i][threadIdx.x] = input[index_in+i*width];
        }

        __syncthreads();

        for (int i=0; i<TILE_DIM; i+=b_row)
        {
            output[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
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

        if (N % TILE_DIM != 0 )
    	{
        printf("Matrix size must be integral multiple of tile size\nExiting...\n\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    	}

    	size_t mem_size = sizeof(float) * N * N;

    	float *input = (float *) malloc(mem_size);
        float *output = (float *) malloc(mem_size);

          // allocate device memory
      
        float *d_input = NULL;
        err = cudaMalloc((void **) &d_input, mem_size);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        float *d_output = NULL;
        err = cudaMalloc((void **) &d_output, mem_size);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        
        for (int i = 0; i < (N*N); ++i)
    	{
        	scanf("%f",&input[i]);
    	}
		
    		/*
		for(long long int i=0;i<N*N; ++i)
		{
			input[i]= (float) i;
		}*/
        // Copy the host input vectors input1 and input2 in host memory to the device input vectors in
        // device memory
        printf("Copy input data from the host memory to the CUDA device\n");
        err = cudaMemcpy(d_input, input, mem_size, cudaMemcpyHostToDevice);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector input1 from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Launch the Vector Add CUDA Kernel
        dim3 threadsPerBlock1(N/TILE_DIM, N/TILE_DIM,1);
        dim3 blocksPerGrid1(TILE_DIM,b_row,1);

        transpose<<<blocksPerGrid1, threadsPerBlock1>>>(d_output, d_input, N, N, 1);
        err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch transpose kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Copy the device result vector in device memory to the host result vector
        // in host memory.
        printf("Copy output data from the CUDA device to the host memory\n");
        err = cudaMemcpy(output, d_output, mem_size, cudaMemcpyDeviceToHost);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector output from device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Verify that the result vector is correct
        printf("Results\n");
        for (long long int i = 0; i < (N*N); ++i)
    	{
        	printf("%0.2f", output[i]);
    	}

        // Free device global memory
        err = cudaFree(d_input);;

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector input1 (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        
        err = cudaFree(d_output);;

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector output1 (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
     
        
        //free(C);
        free(input);
    	free(output);
       
        err = cudaDeviceReset();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    return 0;
}

