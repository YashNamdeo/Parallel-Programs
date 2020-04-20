#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void reduce_kernel(float *a, float *b, int n, int k, int i)
{
    
    int blockNum = blockIdx.y * gridDim.x + blockIdx.x;
    int threadNum = threadIdx.y * blockDim.x + threadIdx.x;
    int j = blockNum * (blockDim.x * blockDim.y) + threadNum;
    
    
    for(unsigned int l = 1; l < k; l *= 2)
    { 
           if (j*i %(2*l*i) == 0 && (j+l)*i < n) 
               a[j*i] += a[(j+l)*i];
           __syncthreads(); 
    }
 
     if((j*i)%k == 0)
        b[j*i]= a[j*i]/k;
    
}

int main(void)
{
    cudaError_t err = cudaSuccess;
    
    int T = 0;
    scanf("%d", &T);
    
    while(T--){
    int p,q;
    scanf("%d %d",&p, &q);

    int n = 2<<(p-1);
    int k = 2<<(q-1);
    size_t size = n*sizeof(float);

    float *h_a = (float*)malloc(size);
    
    if (h_a == NULL)
    {
        fprintf(stderr, "Failed to allocate host array!\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n; ++i)
    scanf("%f", &h_a[i]);

    float *d_a = NULL;
    err = cudaMalloc((void **)&d_a, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device array a (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy array a from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int m;
    int n2=n;
    float *d_b = NULL;
    cudaMalloc((void **)&d_b, size);
    cudaMemset(d_b, 0, n);
    int i=1;
    int nt=(int)(p/q);
    
    while(nt--)
    {
        m = n/k;
        dim3 blocksPerGrid(sqrt(m), sqrt(m), 1);
        dim3 threadsPerBlock(k, 1, 1);
        reduce_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, n2, k, i);
        n = m;
        d_a = d_b;
        
       i*=k;
    }

    float *h_b = (float*)malloc(size);
    cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);
    for(int j=0; j<m; j+=k)
    printf("%.2f ", h_b[j]);

    cudaFree(d_a);
    cudaFree(d_b);
    free(h_b);
    
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return 0;
    }
}