#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <mpi.h>
#include <png.h>
#include <math.h>


void pgmwrite(char *filename, void *vx, int nx, int ny)
{
	FILE *fp;

	int i, j, k, grey;

	float xmin, xmax, tmp, fval;
	float thresh = 255.0;

	float *x = (float *)vx;

	fp = fopen(filename, "w");

	if (NULL == fp)
	{
		printf("pgmwrite: cannot create <%s>\n", filename);
		exit(-1);
	}

	printf("Writing %d x %d picture into file: %s\n", nx, ny, filename);

	xmin = fabs(x[0]);
	xmax = fabs(x[0]);

	for (i = 0; i < nx*ny; i++)
	{
		if (fabs(x[i]) < xmin) xmin = fabs(x[i]);
		if (fabs(x[i]) > xmax) xmax = fabs(x[i]);
	}

	if (xmin == xmax) xmin = xmax - 1.0;

	fprintf(fp, "P2\n");
	fprintf(fp, "# Written by pgmwrite\n");
	fprintf(fp, "%d %d\n", nx, ny);
	fprintf(fp, "%d\n", (int)thresh);

	k = 0;

	for (j = ny - 1; j >= 0; j--)
	{
		for (i = 0; i < nx; i++)
		{

			tmp = x[j + ny * i];

			fval = thresh * ((fabs(tmp) - xmin) / (xmax - xmin)) + 0.5;
			grey = (int)fval;

			fprintf(fp, "%3d ", grey);

			if (0 == (k + 1) % 16) fprintf(fp, "\n");

			k++;
		}
	}

	if (0 != k % 16) fprintf(fp, "\n");
	fclose(fp);
}

void abort_(const char * s, ...)
{
        va_list args;
        va_start(args, s);
        vfprintf(stderr, s, args);
        fprintf(stderr, "\n");
        va_end(args);
        abort();
}

void read_png_file(char* file_name, int *width, int *height, png_bytep **row_pointers)
{
        char header[8]; 

        FILE *fp = fopen(file_name, "rb");

        png_structp png_ptr;
        png_infop info_ptr;
        png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

        if (!png_ptr)
                abort_("[read_png_file] png_create_read_struct failed");

        info_ptr = png_create_info_struct(png_ptr);
        if (!info_ptr)
                abort_("[read_png_file] png_create_info_struct failed");

        if (setjmp(png_jmpbuf(png_ptr)))
                abort_("[read_png_file] Error during init_io");

        png_init_io(png_ptr, fp);
        png_set_sig_bytes(png_ptr, 8);

        png_read_info(png_ptr, info_ptr);

        *width = png_get_image_width(png_ptr, info_ptr);
        *height = png_get_image_height(png_ptr, info_ptr);
        //color_type = png_get_color_type(png_ptr, info_ptr);
        //bit_depth = png_get_bit_depth(png_ptr, info_ptr);

        //number_of_passes = png_set_interlace_handling(png_ptr);
        png_read_update_info(png_ptr, info_ptr);


        /* read file */
        if (setjmp(png_jmpbuf(png_ptr)))
                abort_("[read_png_file] Error during read_image");

        int y;
        *row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * *height);
        for (y=0; y<*height; y++)
                *(row_pointers[y]) = (png_byte*) malloc(png_get_rowbytes(png_ptr,info_ptr));

        png_read_image(png_ptr, *row_pointers);

        fclose(fp);
}

int main(int argc, char *argv[])
{
    int size, rank;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char *filename;
    int width, height;
    png_bytep *row_pointers;
    int masterbuf[width][height];
   

    if(rank == 0)
    {
        filename="input.h";
        read_png_file(filename, &width, &height, &row_pointers);
    }
    int MP = width/size;
    int NP = height;
    float image[MP+2][NP+2];
    int i,j;
    float buf[MP][NP];
    //int hist[256];
    //memset(hist, 0, 256*sizeof(hist[0]));

    /*Histogram Equalization*/
    MPI_Scatter(masterbuf, MP*NP, MPI_FLOAT, buf, MP*NP, MPI_FLOAT, 0, MPI_COMM_WORLD);
    float hist[256];
    memset(hist, 0, 256*sizeof(hist[0]));

    for (i = 1; i < MP + 1; i++)
	{
		for (j = 1; j < NP + 1; j++)
		{
            hist[(int)buf[i][j]]+=1;
		}
	}

    float **histo = (float**)malloc(sizeof(float*)*size);
    for(i=0;i<size;i++)
    histo[i] = (float*)malloc(256*sizeof(float));
    MPI_Allgather(&hist, 1, MPI_FLOAT, histo, 1, MPI_FLOAT, MPI_COMM_WORLD);

    float cumuHist[256];
    memset(cumuHist,0, 256*sizeof(cumuHist[0]));
    memset(hist, 0, 256*sizeof(hist[0]));
    for(i=0;i<256;i++)
    {
        for(j=0;j<size;j++)
        hist[i]+=histo[j][i];       

        if(i!=0)
        cumuHist[i]=cumuHist[i-1]+hist[i]/(width*height);
        else
        {
            cumuHist[i]=hist[i]/(width*height);
        }
    }

    MPI_Scatter(masterbuf, MP*NP, MPI_FLOAT, buf, MP*NP, MPI_FLOAT, 0, MPI_COMM_WORLD);
    for (i = 1; i < MP + 1; i++)
	{
		for (j = 1; j < NP + 1; j++)
		{
			buf[i - 1][j - 1] *=255 * cumuHist[(int)buf[i-1][j-1]];
		}
	} 

	MPI_Gather(buf, MP*NP, MPI_FLOAT, masterbuf, MP*NP, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank ==0)
    {
        char output[]="histeql.pgm";
        filename = output;
        pgmwrite(filename, masterbuf, width, height);
    }

    /* Sobel */
    
    MPI_Scatter(masterbuf, MP*NP, MPI_FLOAT, buf, MP*NP, MPI_FLOAT, 0, MPI_COMM_WORLD);

	for (i = 1; i < MP + 1; i++)
	{
		for (j = 1; j < NP + 1; j++)
		{
			// horizontal gradient
			// -1  0  1
			// -2  0  2
			// -1  0  1

			// vertical gradient
			// -1 -2 -1
			//  0  0  0
			//  1  2  1

			float gradient_h = ((-1.0 * buf[i - 1][j - 1]) + (1.0 * buf[i + 1][j - 1]) + (-2.0 * buf[i - 1][j]) + (2.0 * buf[i + 1][j]) + (-1.0 * buf[i - 1][j + 1]) + (1.0 * buf[i + 1][j + 1]));
			float gradient_v = ((-1.0 * buf[i - 1][j - 1]) + (-2.0 * buf[i][j - 1]) + (-1.0 * buf[i + 1][j - 1]) + (1.0 * buf[i - 1][j + 1]) + (2.0 * buf[i][j + 1]) + (1.0 * buf[i + 1][j + 1]));

			float gradient = sqrt((gradient_h * gradient_h) + (gradient_v * gradient_v));

			if (gradient < 100) {
				gradient = 0;	
			}
			else {
				gradient = 255;	
			}
			image[i][j] = gradient;
		}
	}

	if (rank == 0)
	{
		printf("Finished");
	}

	for (i = 1; i < MP + 1; i++)
	{
		for (j = 1; j < NP + 1; j++)
		{
			buf[i - 1][j - 1] += image[i][j];
		}
	}

	MPI_Gather(buf, MP*NP, MPI_FLOAT, masterbuf, MP*NP, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank ==0)
    {
        char output[]="final.pgm";
        filename = output;
        pgmwrite(filename, masterbuf, width, height);
    }

    MPI_Finalize();
    
    return 0;
}