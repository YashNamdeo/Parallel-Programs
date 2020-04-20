#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include<math.h>

typedef struct points{
    double x,y,z;
}point;

void input(char s[50], point *p, point *q)
{
    
    
    double m[3];
    int j=0, i, flag=0;
    for(i=2; s[i]!=')'; i++)
    {
        char a[20];   
        int k=0;
        while(s[i]!=',')
        {
            if(s[i]==')')
            {
                flag=1;
                break;
            }
            a[k++]=s[i++];
            
        }
        a[k]='\0';
        
        m[j++] = strtof(a, NULL);
        
        if(flag==1)
        break;
    }
    p->x = m[0];
    p->y = m[1];
    p->z = m[2];
    

    flag=0;
    j=0;
    for(i = i+4; s[i]!=')'; i++)
    {   
        char a[20];
        int k=0;
        while(s[i]!=',')
        {
            if(s[i]==')')
            {
                flag=1;
                break;
            }
            a[k++]=s[i++];
            
        }
        a[k]='\0';
        
        m[j++] = strtof(a, NULL);
        
        if(flag==1)
        break;
        
    }

    q->x = m[0];
    q->y = m[1];
    q->z = m[2];
    
}

point* matrixMultiplication(point* object, double matrix[][3], int n)
{
    point* result = (point*)malloc(n*sizeof(point));
    int l;
    #pragma omp for
    for (l=0; l<n; l++)
    {
        result[l].x = result[l].y = result[l].z = 0;
    }

    int i,j,k;
    #pragma omp parallel for schedule(static)
    for (i = 0; i < n; ++i) {
        double A[] = {object[i].x, object[i].y, object[i].z};
        for (j = 0; j < 3; ++j) {
            result[i].x += matrix[0][j] * A[j];
            result[i].y += matrix[1][j] * A[j];
            result[i].z += matrix[2][j] * A[j];
        }
        object[i].x = result[i].x;
        object[i].y = result[i].y;
        object[i].z = result[i].z;
    }
    return object;
}

point* translate(point* object, point p, int n)
{
   
    int i;
    #pragma omp parallel for schedule(static)
    for(i=0; i<n; i++)
    {
        object[i].x = object[i].x - p.x;
        object[i].y = object[i].y - p.y;
        object[i].z = object[i].z - p.z;
    } 
    return object;
}

point* inverseTranslate(point* object, point p, int n)
{
    int i;
    #pragma omp parallel for schedule(dynamic)
    for(i=0; i<n; i++)
    {
        object[i].x = object[i].x + p.x;
        object[i].y = object[i].y + p.y;
        object[i].z = object[i].z + p.z;
    } 
    return object;
}

void normalise(point *q)
{
    double t = sqrt(q->x*q->x + q->y*q->y + q->z*q->z);
    q->x = q->x/t;
    q->y = q->y/t;
    q->z = q->z/t;
}



int main(int argc, char** argv)
{
    omp_set_num_threads(argc);

    FILE *axesFile  = fopen(argv[2], "r");
    FILE *out = fopen("Output.txt", "w");
    if (axesFile  == NULL) 
    {   
        printf("Error! Could not open file\n"); 
        exit(-1); 
    } 
    char s[50];
    
    point p, q;
    if(fgets(s, 50, axesFile)!=NULL)
    {
        input(s, &p, &q);
    }

    FILE *objectFile = fopen(argv[3], "r");
    if (objectFile  == NULL) 
    {   
        printf("Error! Could not open file\n"); 
        exit(-1); 
    }

    point *object;
    double v1, v2, v3;
    int i=0;
    
    object = (point*)malloc(10000*sizeof(point));
    
    while(fscanf(objectFile, "%lf %lf %lf", &v1, &v2, &v3)!=EOF)
    {
        object[i].x = v1;
        object[i].y = v2;
        object[i].z = v3;
        i++;
    }
    
    int numPoints = i;
    
    

    double theta = strtof(argv[4], NULL);
    
    q.x = q.x - p.x;
    q.y = q.y - p.y;
    q.z = q.z - p.z;

    normalise(&q);
    
    double d = sqrt(q.y*q.y + q.z*q.z);
    double rotX [3][3] = {{1, 0, 0}, {0, q.z/d, -1*q.y/d}, {0, q.y/d, q.z/d}};
    double invRotX [3][3] = {{1, 0, 0}, {0, q.z/d, q.y/d}, {0, -1*q.y/d, q.z/d}};

    double rotY[3][3] = {{d, 0, -1*q.x}, {0, 1, 0}, {q.x, 0, d}};
    double invRotY[3][3] = {{d, 0, q.x}, {0, 1, 0}, {-1*q.x, 0, d}};

    double rotZ[3][3] = {{cos(theta), -1*sin(theta), 0}, {sin(theta), cos(theta), 0}, {0, 0, 1}};
    
    
    object = translate(object, p, numPoints);
    
    object = matrixMultiplication(object, rotX, numPoints);
    
    object = matrixMultiplication(object, rotY, numPoints);
    
    object = matrixMultiplication(object, rotZ, numPoints);
    object = matrixMultiplication(object, invRotY, numPoints);
    object = matrixMultiplication(object, invRotX, numPoints);
    object = inverseTranslate(object, p, numPoints);
    
    for(int i=0;i<numPoints;i++)
    {
        fprintf(out, "%.3lf %.3lf %.3lf\n", object[i].x, object[i].y, object[i].z);
    }

    return 0;
}